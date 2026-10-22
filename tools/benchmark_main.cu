#define PT_LIBRARY
#include "../wavefront.cu"
#include <benchmark/benchmark.h>
#include <numeric>

struct Event {
    cudaEvent_t value;
    Event(){CU(cudaEventCreate(&value));}
    ~Event(){cudaEventDestroy(value);}
};

std::vector<Node> sceneNodes(const std::vector<Sphere>& spheres) {
    std::vector<int> ids(spheres.size());std::iota(ids.begin(),ids.end(),0);
    std::vector<Node> nodes;build(nodes,ids,spheres);return nodes;
}

struct RenderFixture {
    std::vector<Sphere> spheres=makeScene();
    std::vector<Node> nodes=sceneNodes(spheres);
    Device<Sphere> ds{spheres.size()};Device<Node> dn{nodes.size()};
    Device<int> error{1},count{1},seen{W*H*SAMPLES},indices{2*W*H*SAMPLES},sizes{2};
    Device<Path> a{W*H*SAMPLES},b{W*H*SAMPLES};Device<Hit> hits{W*H*SAMPLES};
    Device<vec3> colors{W*H*SAMPLES};Device<unsigned> counts{DEPTH};
    Scene scene{ds.p,dn.p,int(spheres.size())};
    RenderBuffers buffers{error.p,count.p,seen.p,indices.p,sizes.p,a.p,b.p,hits.p,colors.p,counts.p};
    Event start,stop;
    std::vector<vec3> reference;
    std::array<unsigned,DEPTH> referenceCounts{};
    uint64_t rays=0;
    bool canPrefetch=false;

    RenderFixture() {
        cudaDeviceProp properties;CU(cudaGetDeviceProperties(&properties,0));canPrefetch=properties.concurrentManagedAccess!=0;
        std::copy(spheres.begin(),spheres.end(),ds.p);std::copy(nodes.begin(),nodes.end(),dn.p);
        *error.p=0;test_intersections(scene,spheres,error.p);
        for(int mode=0;mode<4;mode++) {
            CU(cudaMemset(counts.p,0,DEPTH*sizeof(unsigned)));*error.p=0;
            auto result=renderBackend(scene,buffers,mode,true);
            if(reference.empty()){reference.assign(colors.p,colors.p+W*H*SAMPLES);referenceCounts=result.bounce_counts;}
            require(result.bounce_counts==referenceCounts,"benchmark warmup bounce mismatch");
            checkOutput();
        }
        for(int mode=2;mode<4;mode++) {
            CU(cudaMemset(counts.p,0,DEPTH*sizeof(unsigned)));*error.p=0;
            auto result=renderBackend(scene,buffers,mode,true,false);
            require(result.bounce_counts==referenceCounts,"candidate bounce mismatch");
            checkOutput();
        }
        double energy=0;for(const auto& color:reference)energy+=color.x()+color.y()+color.z();
        require(energy>reference.size()*.001,"benchmark produced black lighting");
        rays=std::accumulate(referenceCounts.begin(),referenceCounts.end(),uint64_t(0));
        resident(ds);resident(dn);resident(a);resident(b);resident(hits);resident(colors);resident(indices);resident(seen);
        synchronize();
    }
    template<class T> void resident(Device<T>& value) {
        if(!canPrefetch)return;
        CU(cudaMemPrefetchAsync(value.p,value.n*sizeof(T),cudaMemLocation{cudaMemLocationTypeDevice,0},0));
    }
    uint64_t bytes() const {
        return ds.n*sizeof(Sphere)+dn.n*sizeof(Node)+(a.n+b.n)*sizeof(Path)+hits.n*sizeof(Hit)+colors.n*sizeof(vec3)+(error.n+count.n+seen.n+indices.n+sizes.n)*sizeof(int)+counts.n*sizeof(unsigned);
    }
    void checkOutput() {
        require(*error.p==0,"GPU queue or traversal error");
        for(size_t i=0;i<reference.size();i++)for(int channel=0;channel<3;channel++)
            require(std::isfinite(colors.p[i][channel])&&std::abs(colors.p[i][channel]-reference[i][channel])<1e-6f,"benchmark linear image oracle mismatch");
    }
};

RenderFixture& sharedFixture(){static RenderFixture fixture;return fixture;}

template<bool Manual,bool WaitAfterRaycast=true> void renderBenchmark(benchmark::State& state) {
    try {
        auto& fixture=sharedFixture();
        fixture.resident(fixture.colors);fixture.resident(fixture.hits);synchronize();
        require(fixture.bytes()<2ull*1024*1024*1024,"fixture exceeds its 2 GiB allocation bound");
        double eventSeconds=0;
        int launches=0;
        for(auto _ : state) {
            state.PauseTiming();
            CU(cudaMemset(fixture.colors.p,0xff,fixture.colors.n*sizeof(vec3)));synchronize();
            state.ResumeTiming();
            CU(cudaEventRecord(fixture.start.value));
            auto result=renderBackend(fixture.scene,fixture.buffers,int(state.range(0)),false,WaitAfterRaycast);
            CU(cudaEventRecord(fixture.stop.value));CU(cudaEventSynchronize(fixture.stop.value));
            float milliseconds=0;CU(cudaEventElapsedTime(&milliseconds,fixture.start.value,fixture.stop.value));
            eventSeconds+=milliseconds/1000.;launches=result.launches;
            if constexpr(Manual)state.SetIterationTime(milliseconds/1000.);
            benchmark::DoNotOptimize(fixture.colors.p);
        }
        fixture.checkOutput();
        state.SetItemsProcessed(state.iterations()*fixture.rays);
        state.counters["managed_bytes"]=double(fixture.bytes());
        state.counters["launches_per_render"]=launches;
        state.counters["event_span_seconds"]=benchmark::Counter(eventSeconds,benchmark::Counter::kAvgIterations);
        state.counters["oracle_pass"]=1;
        state.counters["wait_after_raycast"]=WaitAfterRaycast?1:0;
    } catch(const std::exception& error) {state.SkipWithError(error.what());}
}

void raycastBenchmark(benchmark::State& state) {
    try {
        auto& fixture=sharedFixture();
        fixture.resident(fixture.hits);synchronize();
        initialize<<<(W*H*SAMPLES+127)/128,128>>>(fixture.a.p,fixture.colors.p);synchronize();
        for(auto _ : state) {
            CU(cudaEventRecord(fixture.start.value));
            raycast<<<(W*H*SAMPLES+127)/128,128>>>(fixture.scene,fixture.a.p,fixture.hits.p,W*H*SAMPLES,fixture.error.p);
            CU(cudaGetLastError());CU(cudaEventRecord(fixture.stop.value));CU(cudaEventSynchronize(fixture.stop.value));
            float milliseconds=0;CU(cudaEventElapsedTime(&milliseconds,fixture.start.value,fixture.stop.value));
            state.SetIterationTime(milliseconds/1000.);
        }
        size_t hitCount=0;
        for(int i=0;i<W*H*SAMPLES;i++)hitCount+=fixture.hits.p[i].object>=0;
        require(hitCount>0&&*fixture.error.p==0,"timed raycast did not produce valid hits");
        state.SetItemsProcessed(state.iterations()*uint64_t(W*H*SAMPLES));
        state.counters["result_hits"]=double(hitCount);
        state.counters["oracle_pass"]=1;
    }catch(const std::exception& error){state.SkipWithError(error.what());}
}

int main(int argc,char** argv) {
    try {
        cudaDeviceProp properties;CU(cudaGetDeviceProperties(&properties,0));
        benchmark::AddCustomContext("gpu",properties.name);
        benchmark::AddCustomContext("managed_prefetch",properties.concurrentManagedAccess?"supported":"unsupported; managed-memory migration remains in timing");
        benchmark::AddCustomContext("shape",std::to_string(W)+"x"+std::to_string(H)+" spp="+std::to_string(SAMPLES)+" depth="+std::to_string(DEPTH)+" scene="+std::to_string(PT_SCENE));
        benchmark::AddCustomContext("source_sha256",PT_SOURCE_HASH);
        benchmark::AddCustomContext("adapter_sha256",PT_ADAPTER_HASH);
        benchmark::AddCustomContext("framework","Google Benchmark v1.9.5 commit192ef10025eb2c4cdd392bc502f0c852196baa48");
        benchmark::AddCustomContext("timing_scope","render events include required host scheduling and managed-memory gaps; raycast events bracket one kernel; wall uses framework real time; allocation and oracle checks excluded");
        for(int mode=0;mode<4;mode++) {
            const std::string name=std::array<std::string,4>{"naive","bvh","compact","grouped"}[mode];
            benchmark::RegisterBenchmark((name+"/wall").c_str(),renderBenchmark<false>)->Arg(mode)->UseRealTime()->MinTime(.1)->MinWarmUpTime(.05)->Repetitions(7)->Unit(benchmark::kMillisecond);
            benchmark::RegisterBenchmark((name+"/event_span").c_str(),renderBenchmark<true>)->Arg(mode)->UseManualTime()->MinTime(.1)->MinWarmUpTime(.05)->Repetitions(7)->Unit(benchmark::kMillisecond);
            if(mode>=2) {
                benchmark::RegisterBenchmark((name+"/candidate/wall").c_str(),renderBenchmark<false,false>)->Arg(mode)->UseRealTime()->MinTime(.1)->MinWarmUpTime(.05)->Repetitions(7)->Unit(benchmark::kMillisecond);
                benchmark::RegisterBenchmark((name+"/candidate/event_span").c_str(),renderBenchmark<true,false>)->Arg(mode)->UseManualTime()->MinTime(.1)->MinWarmUpTime(.05)->Repetitions(7)->Unit(benchmark::kMillisecond);
            }
        }
        benchmark::RegisterBenchmark("raycast/kernel_event",raycastBenchmark)->UseManualTime()->MinTime(.1)->MinWarmUpTime(.05)->Repetitions(7)->Unit(benchmark::kMillisecond);
        benchmark::Initialize(&argc,argv);
        if(benchmark::ReportUnrecognizedArguments(argc,argv))return 2;
        benchmark::RunSpecifiedBenchmarks();benchmark::Shutdown();
        return 0;
    }catch(const std::exception& error){std::cerr<<"FAIL "<<error.what()<<'\n';return 1;}
}
