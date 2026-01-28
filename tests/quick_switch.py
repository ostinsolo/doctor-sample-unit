"""Quick model switch test"""
import subprocess, json, time, os

exe = r'C:\Users\soloo\Desktop\shared_runtime\dist\dsu\dsu-bsroformer.exe'
audio = r'C:\Users\soloo\Desktop\shared_runtime\tests\audio\test_mix.wav'
out = r'C:\Users\soloo\Desktop\shared_runtime\tests\quick_test'
model1 = r'C:\Users\soloo\Documents\DSU-VSTOPIA\ThirdPartyApps\Models\bsroformer\weights\bsroformer_4stem.ckpt'
config1 = r'C:\Users\soloo\Desktop\shared_runtime\configs\config_bs_roformer_4stem.yaml'
model2 = r'C:\Users\soloo\Documents\DSU-VSTOPIA\ThirdPartyApps\Models\bsroformer\weights\scnet_xl_ihf.ckpt'
config2 = r'C:\Users\soloo\Desktop\shared_runtime\configs\config_scnet_xl_ihf.yaml'

os.makedirs(out, exist_ok=True)

print('Starting worker...')
p = subprocess.Popen([exe, '--worker'], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, bufsize=1)

def run(cmd):
    p.stdin.write(json.dumps(cmd) + '\n')
    p.stdin.flush()
    while True:
        line = p.stdout.readline()
        if not line: 
            print('  [No response]')
            return None
        try:
            r = json.loads(line.strip())
            s = r.get('status', 'unknown')
            print(f'  [{s}]')
            if s in ['done','error','pong']: 
                return r
        except: 
            pass

print('Ping...')
run({'cmd':'ping'})

print('\nRun 1 (bsroformer COLD)...')
t = time.perf_counter()
r = run({'cmd':'separate','input':audio,'output':out,'model_path':model1,'config_path':config1,'model_type':'bs_roformer'})
elapsed = time.perf_counter()-t
proc = r.get('elapsed', 0) if r else 0
print(f'  Total: {elapsed:.2f}s, Processing: {proc:.2f}s, Overhead: {elapsed-proc:.2f}s')

print('\nRun 2 (bsroformer CACHED)...')
t = time.perf_counter()
r = run({'cmd':'separate','input':audio,'output':out,'model_path':model1,'config_path':config1,'model_type':'bs_roformer'})
elapsed = time.perf_counter()-t
proc = r.get('elapsed', 0) if r else 0
print(f'  Total: {elapsed:.2f}s, Processing: {proc:.2f}s')

print('\nRun 3 (scnet SWITCH)...')
t = time.perf_counter()
r = run({'cmd':'separate','input':audio,'output':out,'model_path':model2,'config_path':config2,'model_type':'scnet'})
elapsed = time.perf_counter()-t
proc = r.get('elapsed', 0) if r else 0
print(f'  Total: {elapsed:.2f}s, Processing: {proc:.2f}s, Switch overhead: {elapsed-proc:.2f}s')

print('\nRun 4 (bsroformer SWITCH BACK)...')
t = time.perf_counter()
r = run({'cmd':'separate','input':audio,'output':out,'model_path':model1,'config_path':config1,'model_type':'bs_roformer'})
elapsed = time.perf_counter()-t
proc = r.get('elapsed', 0) if r else 0
print(f'  Total: {elapsed:.2f}s, Processing: {proc:.2f}s, Switch overhead: {elapsed-proc:.2f}s')

print('\nExit...')
run({'cmd':'exit'})
p.wait(timeout=5)
print('\n=== DONE ===')
