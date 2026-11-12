"""Parameter declarations must match the Config dataclasses of the MRSD package (no Orange needed)."""
import dataclasses

import pytest

from orangecontrib.subgroups import backend, params


@pytest.mark.parametrize("alg", list(params.PARAMS))
def test_params_cover_config_fields(alg):
    fields = backend.config_fields(alg)
    names = [p.name for p in params.PARAMS[alg]]
    assert len(names) == len(set(names)), "duplicate parameter"
    missing = set(fields) - set(names) - set(params.UNSUPPORTED.get(alg, [])) - set(params.IMPLICIT)
    extra = set(names) - set(fields)
    assert not missing, f"config fields without a widget parameter: {missing}"
    assert not extra, f"widget parameters that are not config fields: {extra}"


@pytest.mark.parametrize("alg", list(params.PARAMS))
def test_defaults_build_a_valid_config(alg):
    cfg = params.widget_config(alg, params.setting_defaults(alg))
    backend.config_class(alg)(**cfg)


@pytest.mark.parametrize("alg", list(params.PARAMS))
def test_every_choice_is_accepted(alg):
    base = params.setting_defaults(alg)
    for p in params.PARAMS[alg]:
        if not isinstance(p, (params.ChoiceParam, params.MultiChoiceParam)):
            continue
        for v in p.values:
            s = dict(base)
            s[p.name] = [v] if isinstance(p, params.MultiChoiceParam) else v
            if alg == "ensemble" and p.name == "algorithm_weights" and v == "custom":
                s["custom_weights"] = "pt=1, rules=2"
            backend.config_class(alg)(**params.widget_config(alg, s))


def test_all_algorithms_listed():
    mrsd = backend.load_mrsd()
    assert set(mrsd.ALGORITHMS) == {a.id for a in backend.ALGORITHMS} == set(params.PARAMS)


def test_parsers():
    assert params.parse_int_list(" 2, 3 ;4 ") == [2, 3, 4]
    assert params.parse_int_list("") is None
    assert params.parse_dict("a=1, b:-1", int) == {"a": 1, "b": -1}
    with pytest.raises(ValueError):
        params.parse_dict("nonsense")


def test_parameter_names_do_not_clash_with_widget_attributes():
    """A parameter becomes a widget Setting of the same name; it must not
    shadow an attribute the base widget uses for something else."""
    import os
    import re
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    src = open(os.path.join(here, "orangecontrib", "subgroups", "widgets", "owbase.py"), encoding="utf-8").read()
    attrs = set(re.findall(r"self\.([A-Za-z_]\w*)", src)) | set(re.findall(r"^    (\w+) = ", src, re.M)) \
        | set(re.findall(r"    def (\w+)\(", src))
    names = set()
    for alg in params.PARAMS:
        names |= set(params.setting_defaults(alg))
    assert not (names & attrs), f"clashing names: {sorted(names & attrs)}"
