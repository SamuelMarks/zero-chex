import sys
import os
import json
import inspect

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
)
import zero_chex


def main():
    with open("chex_api_snapshot.json", "r") as f:
        chex_api = json.load(f)

    zero_chex_all = set(getattr(zero_chex, "__all__", []))
    chex_all = set(chex_api.keys())

    # We allow get_err_regex because zero_chex implements it.
    zero_chex_all.discard("get_err_regex")

    missing_in_zero = chex_all - zero_chex_all
    extra_in_zero = zero_chex_all - chex_all

    errors = []
    if missing_in_zero:
        errors.append(f"Missing in zero_chex: {sorted(missing_in_zero)}")
    if extra_in_zero:
        errors.append(f"Extra in zero_chex: {sorted(extra_in_zero)}")

    # We could also check function signatures
    # Let's see if there are any signature parameter name mismatches.
    for name in chex_all.intersection(zero_chex_all):
        chex_info = chex_api[name]
        obj = getattr(zero_chex, name)

        if chex_info["kind"] == "function" and inspect.isroutine(obj):
            try:
                sig_zero = inspect.signature(obj)
                # just check parameter names
                zero_params = list(sig_zero.parameters.keys())

                import chex

                chex_obj = getattr(chex, name)
                try:
                    chex_sig = inspect.signature(chex_obj)
                    chex_params = list(chex_sig.parameters.keys())
                    if chex_params != zero_params:
                        errors.append(
                            f"{name} parameter mismatch: chex={chex_params}, zero={zero_params}"
                        )
                except ValueError:
                    pass
            except ValueError:
                pass

    if errors:
        print("API Parity Check Failed:")
        for err in errors:
            print(" -", err)
        sys.exit(1)
    else:
        print("API Parity Check Passed!")
        sys.exit(0)


if __name__ == "__main__":
    main()
