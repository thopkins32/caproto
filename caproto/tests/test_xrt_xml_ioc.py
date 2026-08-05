import pytest

from caproto.ioc_examples.xrt_xml_ioc import XmlEntry, _unique_suffixes


def test_unique_suffixes_drop_xml_contexts():
    entries = [
        XmlEntry(
            path=(
                "Project",
                "myTestBeamline",
                "toroidMirror01",
                "properties",
                "precisionOpenCL",
            ),
            raw_text="float64",
            value="float64",
        ),
        XmlEntry(
            path=(
                "Project",
                "myTestBeamline",
                "toroidMirror01",
                "reflect",
                "parameters",
                "needLocal",
            ),
            raw_text="True",
            value=True,
        ),
        XmlEntry(
            path=("Project", "Materials", "crystalSi01", "properties", "tK"),
            raw_text="297.15",
            value=297.15,
        ),
        XmlEntry(
            path=("Project", "FigureErrors", "waviness01", "properties", "amplitude"),
            raw_text="10.0",
            value=10.0,
        ),
    ]

    assert _unique_suffixes(
        entries,
        drop_top_level={"myTestBeamline", "Materials", "FigureErrors"},
    ) == [
        "toroidMirror01:precisionOpenCL",
        "toroidMirror01:reflect:needLocal",
        "crystalSi01:tK",
        "waviness01:amplitude",
    ]


def test_unique_suffixes_fall_back_on_shortened_collision():
    entries = [
        XmlEntry(
            path=("Project", "BMM", "Si111", "properties", "rho"),
            raw_text="2.33",
            value=2.33,
        ),
        XmlEntry(
            path=("Project", "Materials", "Si111", "properties", "rho"),
            raw_text="0",
            value=0,
        ),
    ]

    assert _unique_suffixes(
        entries,
        drop_top_level={"BMM", "Materials", "FigureErrors"},
    ) == [
        "BMM:Si111:rho",
        "Materials:Si111:rho",
    ]


def test_unique_suffixes_drop_method_context_for_length_limit():
    entries = [
        XmlEntry(
            path=(
                "Project",
                "myTestBeamline",
                "toroidMirror01",
                "reflect",
                "parameters",
                "beam",
            ),
            raw_text="bendingMagnet01_global",
            value="bendingMagnet01_global",
        ),
        XmlEntry(
            path=(
                "Project",
                "myTestBeamline",
                "toroidMirror01",
                "reflect",
                "parameters",
                "noIntersectionSearch",
            ),
            raw_text="False",
            value=False,
        ),
    ]

    assert _unique_suffixes(
        entries,
        drop_top_level={"myTestBeamline", "Materials", "FigureErrors"},
        max_length=39,
    ) == [
        "toroidMirror01:reflect:beam",
        "toroidMirror01:noIntersectionSearch",
    ]


def test_unique_suffixes_raise_when_no_candidate_fits_length_limit():
    entry = XmlEntry(
        path=(
            "Project",
            "myTestBeamline",
            "toroidMirror01",
            "reflect",
            "parameters",
            "noIntersectionSearch",
        ),
        raw_text="False",
        value=False,
    )

    with pytest.raises(ValueError, match="within 10 characters"):
        _unique_suffixes(
            [entry],
            drop_top_level={"myTestBeamline", "Materials", "FigureErrors"},
            max_length=10,
        )


@pytest.mark.parametrize(
    "entry, expected",
    [
        (
            XmlEntry(
                path=("Project", "BMM", "DCM", "properties", "center"),
                raw_text="[0, 26105, 'auto']",
                value=0,
                field_name="x",
                field_index=0,
            ),
            "DCM:center:x",
        ),
        (
            XmlEntry(
                path=("Project", "BMM", "DCM", "double_reflect", "parameters", "beam"),
                raw_text="M1_VCM_global",
                value="M1_VCM_global",
            ),
            "DCM:double_reflect:beam",
        ),
    ],
)
def test_unique_suffixes_keep_required_detail(entry, expected):
    assert _unique_suffixes(
        [entry],
        drop_top_level={"BMM", "Materials", "FigureErrors"},
    ) == [expected]
