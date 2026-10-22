"""Check retained GPU proof artifacts on CPU; this does not execute CUDA."""
from pathlib import Path
import re
import subprocess

root = Path(__file__).resolve().parents[1]
names = ['naive-megakernel', 'bvh-megakernel', 'compact-wavefront', 'grouped-wavefront']
images = []
for name in names:
    tokens = (root / 'evidence' / (name + '.ppm')).read_text().split()
    assert tokens[:4] == ['P3', '96', '64', '255'], name
    pixels = list(map(int, tokens[4:]))
    assert len(pixels) == 96 * 64 * 3
    assert all(0 <= x <= 255 for x in pixels) and sum(pixels) > 100
    images.append(pixels)
assert all(image == images[0] for image in images)
raw_log = (root / 'evidence/verification.txt').read_bytes()
log = raw_log.decode('utf-16' if raw_log.startswith(b'\xff\xfe') else 'utf-8-sig')
assert 'intersection_tests=4102 GPU_BVH_naive_CPU=pass' in log
assert 'PASS finite_lighting parity unique_ids bounded_queues cleanup' in log
assert len(re.findall(r'raycast_kernel repeat=\d event_ms=[0-9.]+ rays=24576 hits=\d+', log)) == 3
for name in names:
    rows = re.findall(r'backend=' + name + r' event_interval_ms=([0-9.]+) kernel_launches=(\d+) max_error=0 rays=([0-9,]+)', log)
    assert len(rows) == 5, name
    assert all(float(ms) > 0 and int(launches) > 0 for ms, launches, counts in rows)
    assert all(counts == rows[0][2] for ms, launches, counts in rows)
for file in ['main.cu', 'vec3.h', 'ray.h', 'sphere.h', 'camera.h', 'material.h', 'hitable.h', 'hitable_list.h', 'Makefile']:
    original = subprocess.check_output(['git', 'show', 'ab140b12d4923b75270831baabab5e4d4209f305:' + file], cwd=root)
    assert original.replace(b'\r\n', b'\n') == (root / file).read_bytes().replace(b'\r\n', b'\n'), file
print('PASS retained images, timing records, and unchanged upstream source; GPU execution is historical')
