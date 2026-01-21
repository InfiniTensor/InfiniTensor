import sys

sys.path.extend(__path__)

if 'backend' in sys.modules:
    backend = sys.modules['backend']
else:
    import backend

print("import backend: {}".format(backend))
