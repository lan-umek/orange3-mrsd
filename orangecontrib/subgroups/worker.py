"""
Worker process: runs one MRSD algorithm outside the Orange process.

    python -m orangecontrib.subgroups.worker <input.pkl> <output.pkl>

The input pickle holds the keyword arguments of ``backend.run_algorithm``;
the output pickle holds ("ok", SubgroupResults) or ("error", message).
Running in a separate process means that (a) *Stop* really stops the
computation (the process is killed), (b) a long run does not slow down the
other widgets or Orange's interface (no shared interpreter lock).
"""
import pickle
import sys
import traceback


def main(argv):
    inp, out = argv[1], argv[2]
    print("worker: started with", sys.executable, flush=True)
    try:
        from orangecontrib.subgroups import backend
        print("worker: add-on imported from", backend.__file__, flush=True)
        backend.load_mrsd()
        with open(inp, "rb") as f:
            kwargs = pickle.load(f)
        print("worker: running", kwargs.get("alg_id"), flush=True)
        result = backend.run_algorithm(**kwargs)
        print("worker: done,", len(result), "subgroups", flush=True)
        result.data = None
        payload = ("ok", result)
    except BaseException as ex:      # noqa: BLE001 - reported to the parent
        payload = ("error", f"{type(ex).__name__}: {ex}\n{traceback.format_exc()}")
    with open(out, "wb") as f:
        pickle.dump(payload, f)


if __name__ == "__main__":
    main(sys.argv)
