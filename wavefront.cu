#include <cuda_runtime.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>
#include "vec3.h"

#define HD __host__ __device__
void checked(cudaError_t e, const char* where) {
    if (e != cudaSuccess) throw std::runtime_error(std::string(where)+": "+cudaGetErrorString(e));
}
#define CU(x) checked((x), #x)
constexpr int W=96,H=64,SAMPLES=4,DEPTH=8,MAX_SPHERES=128,STACK=32;
struct Ray { vec3 o,d; };
struct Sphere { vec3 c,albedo; float r; int material; };
struct Node { vec3 lo,hi; int left,right,object; };
struct Hit { float t; int object; vec3 n; };
struct Path { Ray ray; vec3 throughput; int id; };
struct Scene { Sphere* spheres; Node* nodes; int count; };
template<class T> struct Device {
    T* p=nullptr; size_t n;
    explicit Device(size_t count):n(count) { CU(cudaMallocManaged(&p,n*sizeof(T))); }
    ~Device(){ if(p) cudaFree(p); }
    Device(const Device&)=delete;
};
HD float component(vec3 v,int a){return v[a];}
HD bool sphere_hit(Ray r,Sphere s,float maximum,Hit& hit,int object) {
    vec3 oc=r.o-s.c; float a=dot(r.d,r.d),b=dot(oc,r.d),c=dot(oc,oc)-s.r*s.r;
    float disc=b*b-a*c; if(disc<0) return false;
    float root=(-b-sqrtf(disc))/a;
    if(root<0.001f || root>maximum){root=(-b+sqrtf(disc))/a;if(root<0.001f||root>maximum)return false;}
    hit={root,object,(r.o+root*r.d-s.c)/s.r};return true;
}
HD bool box_hit(Ray r,Node n,float maximum) {
    float near=0.001f,far=maximum;
    for(int a=0;a<3;a++) {
        if(fabsf(r.d[a])<1e-20f) {if(r.o[a]<n.lo[a]||r.o[a]>n.hi[a])return false;continue;}
        float x=(n.lo[a]-r.o[a])/r.d[a],y=(n.hi[a]-r.o[a])/r.d[a];
        if(x>y){float z=x;x=y;y=z;} near=fmaxf(near,x);far=fminf(far,y);if(far<near)return false;
    }return true;
}
__device__ Hit intersect(Ray ray,Scene scene,bool bvh,int* error) {
    Hit best={1e30f,-1,vec3(0,0,0)};
    if(!bvh){for(int i=0;i<scene.count;i++)sphere_hit(ray,scene.spheres[i],best.t,best,i);return best;}
    int stack[STACK],size=1;stack[0]=0;
    while(size) {
        Node node=scene.nodes[stack[--size]];if(!box_hit(ray,node,best.t))continue;
        if(node.object>=0){sphere_hit(ray,scene.spheres[node.object],best.t,best,node.object);continue;}
        if(size+2>STACK){atomicExch(error,1);return best;}
        stack[size++]=node.left;stack[size++]=node.right;
    }return best;
}
HD unsigned mix(unsigned x){x^=x>>16;x*=0x7feb352du;x^=x>>15;x*=0x846ca68bu;return x^(x>>16);}
HD float random01(int id,int bounce,int dimension){return (mix(unsigned(id)*747796405u+unsigned(bounce+1)*2891336453u+unsigned(dimension)*277803737u)>>8)*0x1p-24f;}
__device__ Path primary(int id){
    int pixel=id/SAMPLES,x=pixel%W,y=pixel/W;
    float u=(x+random01(id,-1,0))/W,v=(y+random01(id,-1,1))/H;
    return {{vec3(0,1.2f,5),unit_vector(vec3((u-.5f)*3,(v-.5f)*2-.15f,-3))},vec3(1,1,1),id};
}
__device__ vec3 sky(vec3 d){float t=.5f*(unit_vector(d).y()+1);return (1-t)*vec3(1,1,1)+t*vec3(.45f,.65f,1);}
template<int MATERIAL> __device__ void scatter(Path& p,Hit h,Scene s,int bounce) {
    Sphere object=s.spheres[h.object];vec3 point=p.ray.o+h.t*p.ray.d;
    if constexpr(MATERIAL==1) {
        p.ray={point+.002f*h.n,unit_vector(p.ray.d-2*dot(p.ray.d,h.n)*h.n)};
    } else {
        float phi=6.28318530718f*random01(p.id,bounce,0),r=sqrtf(random01(p.id,bounce,1));
        vec3 axis=fabsf(h.n.x())>.9f?vec3(0,1,0):vec3(1,0,0);
        vec3 tangent=unit_vector(cross(axis,h.n)),bitangent=cross(h.n,tangent);
        vec3 direction=r*cosf(phi)*tangent+r*sinf(phi)*bitangent+sqrtf(fmaxf(0.f,1-r*r))*h.n;
        p.ray={point+.002f*h.n,unit_vector(direction)};
    }
    p.throughput*=object.albedo;
}
__global__ void megakernel(Scene s,vec3* colors,bool bvh,int* error,unsigned* counts) {
    int id=blockIdx.x*blockDim.x+threadIdx.x;if(id>=W*H*SAMPLES)return;
    Path p=primary(id);colors[id]=vec3(0,0,0);
    for(int bounce=0;bounce<DEPTH;bounce++) {
        atomicAdd(counts+bounce,1);Hit h=intersect(p.ray,s,bvh,error);
        if(h.object<0){colors[id]=p.throughput*sky(p.ray.d);return;}
        if(s.spheres[h.object].material==1)scatter<1>(p,h,s,bounce);else scatter<0>(p,h,s,bounce);
    }
}
__global__ void initialize(Path* paths,vec3* colors){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<W*H*SAMPLES){paths[i]=primary(i);colors[i]=vec3(0,0,0);}}
__global__ void raycast(Scene s,Path* paths,Hit* hits,int n,int* error){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n)hits[i]=intersect(paths[i].ray,s,true,error);}
__global__ void group_materials(Scene s,Path* paths,Hit* hits,int n,int* indices,int* sizes,vec3* colors){
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n)return;
    if(hits[i].object<0){colors[paths[i].id]=paths[i].throughput*sky(paths[i].ray.d);return;}
    int material=s.spheres[hits[i].object].material;
    int slot=atomicAdd(sizes+material,1);indices[material*W*H*SAMPLES+slot]=i;
}
// Every survivor reserves exactly one output slot; each queue has the full input capacity.
template<int MATERIAL> __global__ void shade(Scene s,Path* in,Hit* hits,int n,Path* out,int* count,vec3* colors,int bounce,int* error,int* indices=nullptr) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n)return;if(indices)i=indices[i];Path p=in[i];Hit h=hits[i];
    if(h.object<0){if constexpr(MATERIAL<=0)colors[p.id]=p.throughput*sky(p.ray.d);return;}
    int material=s.spheres[h.object].material;
    if constexpr(MATERIAL>=0){if(material!=MATERIAL)return;}
    if constexpr(MATERIAL==1)scatter<1>(p,h,s,bounce);
    else if constexpr(MATERIAL==0)scatter<0>(p,h,s,bounce);
    else {if(material==1)scatter<1>(p,h,s,bounce);else scatter<0>(p,h,s,bounce);}
    int slot=atomicAdd(count,1);if(slot>=W*H*SAMPLES){atomicExch(error,2);return;}out[slot]=p;
}
__global__ void check_ids(Path* p,int n,int* seen,int* error){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n){int id=p[i].id;if(id<0||id>=W*H*SAMPLES)atomicExch(error,3);else if(atomicAdd(seen+id,1))atomicExch(error,4);}}
__global__ void intersection_tests(Scene s,Ray* rays,Hit* naive,Hit* tree,int n,int* error){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n){naive[i]=intersect(rays[i],s,false,error);tree[i]=intersect(rays[i],s,true,error);}}
vec3 low(vec3 a,vec3 b){return vec3(std::min(a.x(),b.x()),std::min(a.y(),b.y()),std::min(a.z(),b.z()));}
vec3 high(vec3 a,vec3 b){return vec3(std::max(a.x(),b.x()),std::max(a.y(),b.y()),std::max(a.z(),b.z()));}
int build(std::vector<Node>& nodes,std::vector<int> ids,const std::vector<Sphere>& spheres,int depth=0){
    if(depth>=STACK-2)throw std::runtime_error("BVH depth exceeds traversal stack");
    int index=int(nodes.size());nodes.push_back({});vec3 lo(1e30f,1e30f,1e30f),hi(-1e30f,-1e30f,-1e30f);
    for(int id:ids){const auto& s=spheres[id];vec3 r(s.r,s.r,s.r);lo=low(lo,s.c-r);hi=high(hi,s.c+r);}
    if(ids.size()==1){nodes[index]={lo,hi,-1,-1,ids[0]};return index;}
    vec3 extent=hi-lo;int axis=extent.x()>extent.y()?0:1;if(extent.z()>extent[axis])axis=2;
    std::sort(ids.begin(),ids.end(),[&](int a,int b){return spheres[a].c[axis]<spheres[b].c[axis];});
    auto middle=ids.begin()+ids.size()/2;int left=build(nodes,{ids.begin(),middle},spheres,depth+1),right=build(nodes,{middle,ids.end()},spheres,depth+1);
    nodes[index]={lo,hi,left,right,-1};return index;
}
void synchronize(){CU(cudaGetLastError());CU(cudaDeviceSynchronize());}
void require(bool condition,const char* message){if(!condition)throw std::runtime_error(message);}
// Independent double-precision CPU quadratic oracle, separate from the GPU sphere function.
double oracle(Ray ray,const std::vector<Sphere>& spheres){double best=1e30;for(const auto& s:spheres){double a=0,b=0,c=-double(s.r)*s.r;for(int k=0;k<3;k++){double o=double(ray.o[k])-s.c[k],d=ray.d[k];a+=d*d;b+=2*o*d;c+=o*o;}double disc=b*b-4*a*c;if(disc<0)continue;for(double t:{(-b-std::sqrt(disc))/(2*a),(-b+std::sqrt(disc))/(2*a)})if(t>=.001&&t<best)best=t;}return best;}
void test_intersections(Scene scene,const std::vector<Sphere>& spheres,int* error){
    std::vector<Ray> test={{{0,1,5},{0,0,-1}},{{0,0,0},{1,0,0}},{{0,1,0},{0,1,0}},{{100,100,100},{1,0,0}},{{-1.5f,.7f,0},{0,0,1}},{{-1.5f,1.4f,4},{0,0,-1}}};
    for(int i=0;i<4096;i++){vec3 o((random01(i,0,0)-.5f)*10,random01(i,0,1)*5,(random01(i,0,2)-.5f)*10),d(random01(i,1,0)-.5f,random01(i,1,1)-.5f,random01(i,1,2)-.5f);test.push_back({o,unit_vector(d)});}
    Device<Ray> rays(test.size());Device<Hit> naive(test.size()),tree(test.size());std::copy(test.begin(),test.end(),rays.p);
    intersection_tests<<<(int(test.size())+127)/128,128>>>(scene,rays.p,naive.p,tree.p,int(test.size()),error);synchronize();
    for(size_t i=0;i<test.size();i++){require(naive.p[i].object==tree.p[i].object,"BVH object mismatch");require(fabsf(naive.p[i].t-tree.p[i].t)<1e-5f,"BVH distance mismatch");double t=oracle(test[i],spheres);if(t<1e29)require(std::abs(tree.p[i].t-t)<.003*std::max(1.,t),"CPU oracle mismatch");else require(tree.p[i].object<0,"CPU miss mismatch");}
    require(*error==0,"GPU intersection error");std::cout<<"intersection_tests="<<test.size()<<" GPU_BVH_naive_CPU=pass\n";
}
void image(const std::filesystem::path& file,const std::vector<vec3>& values){std::ofstream out(file);require(bool(out),"image output open failed");out<<"P3\n"<<W<<' '<<H<<"\n255\n";double energy=0;for(int y=H-1;y>=0;y--)for(int x=0;x<W;x++){vec3 c(0,0,0);for(int sample=0;sample<SAMPLES;sample++)c+=values[(y*W+x)*SAMPLES+sample];c/=float(SAMPLES);for(int k=0;k<3;k++){require(std::isfinite(c[k])&&c[k]>=0&&c[k]<=1.001f,"invalid radiance");energy+=c[k];out<<int(255.99f*sqrtf(std::min(1.f,c[k])))<<' ';}out<<'\n';}require(energy>100,"black render");require(bool(out),"image write failed");}
int main(int argc,char** argv){try{
    require(argc<=2,"usage: path-tracer [output-directory|--doctor]");cudaDeviceProp prop;CU(cudaGetDeviceProperties(&prop,0));std::cout<<"GPU="<<prop.name<<" compute="<<prop.major<<'.'<<prop.minor<<"\n";
    if(argc==2&&std::string(argv[1])=="--doctor")return 0;
    std::filesystem::path directory=argc==2?argv[1]:"evidence";std::filesystem::create_directories(directory);
    std::vector<Sphere> spheres={{{0,-1000,0},{.65f,.65f,.65f},1000,0},{{-1.5f,.7f,0},{.8f,.25f,.15f},.7f,0},{{0,.7f,0},{.8f,.85f,.9f},.7f,1},{{1.5f,.7f,0},{.2f,.6f,.3f},.7f,0}};
    for(int x=-4;x<=4;x++)for(int z=-5;z<=-2;z++)spheres.push_back({vec3(x*.65f,.2f,z*.65f),vec3(.3f+.04f*(x+4),.4f,.65f),.2f,(x+z)%2==0?1:0});
    require(spheres.size()<=MAX_SPHERES,"scene exceeds sphere capacity");
    for(auto s:spheres){require(s.r>0&&std::isfinite(s.r)&&s.material>=0&&s.material<2,"invalid scene");for(int k=0;k<3;k++)require(std::isfinite(s.c[k])&&std::abs(s.c[k])<=1000&&std::isfinite(s.albedo[k])&&s.albedo[k]>=0&&s.albedo[k]<=1,"invalid geometry or material");}
    std::vector<int> ids;for(int i=0;i<int(spheres.size());i++)ids.push_back(i);std::vector<Node> nodes;build(nodes,ids,spheres);
    Device<Sphere> ds(spheres.size());Device<Node> dn(nodes.size());std::copy(spheres.begin(),spheres.end(),ds.p);std::copy(nodes.begin(),nodes.end(),dn.p);Scene scene{ds.p,dn.p,int(spheres.size())};
    Device<int> error(1),count(1),seen(W*H*SAMPLES),material_indices(2*W*H*SAMPLES),material_sizes(2);*error.p=0;test_intersections(scene,spheres,error.p);
    Device<Path> a(W*H*SAMPLES),b(W*H*SAMPLES);Device<Hit> hits(W*H*SAMPLES);Device<vec3> colors(W*H*SAMPLES);Device<unsigned> counts(DEPTH);
    std::array<std::string,4> names={"naive-megakernel","bvh-megakernel","compact-wavefront","grouped-wavefront"};std::vector<vec3> reference;std::array<unsigned,DEPTH> reference_counts{};
    cudaEvent_t start,stop;CU(cudaEventCreate(&start));CU(cudaEventCreate(&stop));
    initialize<<<(W*H*SAMPLES+127)/128,128>>>(a.p,colors.p);synchronize();
    for(int repeat=0;repeat<3;repeat++){
        CU(cudaEventRecord(start));
        raycast<<<(W*H*SAMPLES+127)/128,128>>>(scene,a.p,hits.p,W*H*SAMPLES,error.p);
        CU(cudaGetLastError());CU(cudaEventRecord(stop));CU(cudaEventSynchronize(stop));
        float ms;CU(cudaEventElapsedTime(&ms,start,stop));int hit_count=0;
        for(int i=0;i<W*H*SAMPLES;i++)hit_count+=hits.p[i].object>=0;
        require(hit_count>0&&*error.p==0,"raycast timing work missing");
        std::cout<<"raycast_kernel repeat="<<repeat<<" event_ms="<<ms<<" rays="<<W*H*SAMPLES<<" hits="<<hit_count<<"\n";
    }
    for(int repeat=-1;repeat<5;repeat++)for(int mode=0;mode<4;mode++){
        CU(cudaMemset(counts.p,0,DEPTH*sizeof(unsigned)));*error.p=0;int launches=0;std::array<unsigned,DEPTH> bounce_counts{};CU(cudaEventRecord(start));
        if(mode<2){megakernel<<<(W*H*SAMPLES+127)/128,128>>>(scene,colors.p,mode==1,error.p,counts.p);launches++;synchronize();std::copy(counts.p,counts.p+DEPTH,bounce_counts.begin());}
        else {initialize<<<(W*H*SAMPLES+127)/128,128>>>(a.p,colors.p);launches++;synchronize();Path* in=a.p;Path* out=b.p;int n=W*H*SAMPLES;
            for(int bounce=0;bounce<DEPTH;bounce++) {bounce_counts[bounce]=n;if(!n)break;*count.p=0;
                raycast<<<(n+127)/128,128>>>(scene,in,hits.p,n,error.p);launches++;synchronize();
                if(mode==2){shade<-1><<<(n+127)/128,128>>>(scene,in,hits.p,n,out,count.p,colors.p,bounce,error.p);launches++;}
                else{CU(cudaMemset(material_sizes.p,0,2*sizeof(int)));group_materials<<<(n+127)/128,128>>>(scene,in,hits.p,n,material_indices.p,material_sizes.p,colors.p);launches++;synchronize();
                    require(material_sizes.p[0]+material_sizes.p[1]<=n,"material queue capacity");
                    int diffuse=material_sizes.p[0],metal=material_sizes.p[1];
                    if(diffuse){shade<0><<<(diffuse+127)/128,128>>>(scene,in,hits.p,diffuse,out,count.p,colors.p,bounce,error.p,material_indices.p);launches++;CU(cudaGetLastError());}
                    if(metal){shade<1><<<(metal+127)/128,128>>>(scene,in,hits.p,metal,out,count.p,colors.p,bounce,error.p,material_indices.p+W*H*SAMPLES);launches++;}}
                synchronize();require(*count.p>=0&&*count.p<=n,"lost queue capacity invariant");n=*count.p;
                CU(cudaMemset(seen.p,0,W*H*SAMPLES*sizeof(int)));if(n){check_ids<<<(n+127)/128,128>>>(out,n,seen.p,error.p);launches++;synchronize();}std::swap(in,out);
            }
        }
        CU(cudaEventRecord(stop));CU(cudaEventSynchronize(stop));float ms;CU(cudaEventElapsedTime(&ms,start,stop));require(*error.p==0,"GPU queue or traversal failure");
        std::vector<vec3> result(colors.p,colors.p+W*H*SAMPLES);
        if(reference.empty()){reference=result;reference_counts=bounce_counts;}
        require(bounce_counts==reference_counts,"lost or extra paths at bounce");float max_error=0;for(size_t i=0;i<result.size();i++)for(int k=0;k<3;k++)max_error=std::max(max_error,std::abs(result[i][k]-reference[i][k]));require(max_error<1e-6f,"integrator parity mismatch");
        if(repeat>=0){std::cout<<"repeat="<<repeat<<" backend="<<names[mode]<<" event_interval_ms="<<ms<<" kernel_launches="<<launches<<" max_error="<<max_error<<" rays=";for(auto c:bounce_counts)std::cout<<c<<',';std::cout<<"\n";}
        if(repeat==0)image(directory/(names[mode]+".ppm"),result);
    }
    CU(cudaEventDestroy(start));CU(cudaEventDestroy(stop));std::cout<<"PASS finite_lighting parity unique_ids bounded_queues cleanup\n";return 0;
}catch(const std::exception& e){std::cerr<<"FAIL "<<e.what()<<'\n';return 1;}}
