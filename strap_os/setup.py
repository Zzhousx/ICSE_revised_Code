from pathlib import Path
import re, subprocess, sys, urllib.request

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs'
COMMIT = '1ba86ff54e77ff61548f458ea7559578564bce70'

def main():
    OUT.mkdir(exist_ok=True)
    clone = OUT / 'strap'
    if not clone.exists():
        subprocess.run(['git', '-c', 'core.autocrlf=false', 'clone', 'https://github.com/ITSEG-MQ/STRAP.git', str(clone)], check=True)
    subprocess.run(['git', '-C', str(clone), 'checkout', '--detach', COMMIT], check=True)
    actual = subprocess.check_output(['git', '-C', str(clone), 'rev-parse', 'HEAD'], text=True).strip()
    if actual != COMMIT:
        raise RuntimeError('The STRaP checkout does not match the pinned commit.')
    base = 'https://raw.githubusercontent.com/ApolloAuto/apollo/r5.0.0/'
    protos = OUT / 'proto'; generated = OUT / 'generated'
    pending = ['modules/map/proto/map.proto']; seen = set()
    while pending:
        name = pending.pop()
        if name in seen:
            continue
        seen.add(name)
        path = protos / name
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            with urllib.request.urlopen(base + name, timeout=60) as response:
                path.write_bytes(response.read())
        pending.extend(re.findall(r'import\s+"([^"]+)"', path.read_text(encoding='utf-8')))
    generated.mkdir(exist_ok=True)
    subprocess.run([sys.executable, '-m', 'grpc_tools.protoc', '-I' + str(protos),
                    '--python_out=' + str(generated), *(str(protos / p) for p in sorted(seen))], check=True)
    print('STRaP and Apollo protocol schemas are ready in outputs/.')

if __name__ == '__main__':
    main()
