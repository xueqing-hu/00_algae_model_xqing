from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path
from typing import List, Optional, Tuple


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def sanitize_filename(name: str) -> str:
    sanitized = re.sub(r"[^A-Za-z0-9._\-]+", "_", name.strip())
    sanitized = re.sub(r"_+", "_", sanitized).strip("._")
    return sanitized or "material"


def extract_brick_type(material_name: str) -> str:
    """Extract brick type code (ZA-ZQ) from material name, e.g. 'Old Building Brick Dresden ZG [495]' -> 'ZG'."""
    match = re.search(r'Z([A-Qa-q])', material_name)
    if match:
        return f"Z{match.group(1).upper()}"
    return sanitize_filename(material_name)


def discover_materials(template_text: str) -> List[str]:
    materials = re.findall(r'<MaterialReference name="([^"]+)"', template_text)
    return list(dict.fromkeys(materials))


def discover_materials_from_folder(materials_dir: Path) -> List[str]:
    """Return sorted list of .m6 file stems from the given directory."""
    stems = sorted(p.stem for p in materials_dir.glob("*.m6"))
    if not stems:
        raise RuntimeError(f"No .m6 files found in {materials_dir}")
    return stems


def replace_single_parameter(template_text: str, parameter_name: str, new_value: str) -> str:
    pattern = (
        rf'(<IBK:Parameter\s+name="{re.escape(parameter_name)}"\s+unit="[^"]*">)'
        rf'([^<]+)'
        rf'(</IBK:Parameter>)'
    )
    replaced_text, n = re.subn(pattern, rf'\g<1>{new_value}\g<3>', template_text, count=1)
    if n == 0:
        raise RuntimeError(f"Could not find parameter '{parameter_name}' in template.")
    return replaced_text


def replace_start_year(template_text: str, new_start_year: int) -> str:
    pattern = r'(<StartYear>)([^<]+)(</StartYear>)'
    replaced_text, n = re.subn(pattern, rf'\g<1>{new_start_year}\g<3>', template_text, count=1)
    if n == 0:
        raise RuntimeError("Could not find <StartYear>...</StartYear> in template.")
    return replaced_text


def build_variant_text(
    template_text: str,
    materials_dir_for_delphin: str,
    climate_file_for_delphin: str,
    re_value: float,
    orientation_value: int,
    solar_value: float,
    material_name: str,
    start_year: Optional[int],
    duration_years: Optional[float],
) -> str:
    variant = template_text

    # Material paths
    variant = variant.replace("${Project Directory}/Delphin_Brick", materials_dir_for_delphin)
    variant = variant.replace(
        "C:/Program Files/IBK/Delphin 6.1/resources/DB_materials",
        materials_dir_for_delphin,
    )

    # Climate file path
    variant = variant.replace(
        "${Project Directory}/KMI_Brussels_1987-2017_4CY.wac",
        climate_file_for_delphin,
    )

    # Template placeholders
    variant = variant.replace("${RE_VALUE}", f"{re_value}")
    variant = variant.replace("${ORIENTATION_VALUE}", f"{orientation_value}")
    variant = variant.replace("${SOLAR_VALUE}", f"{solar_value}")
    variant = variant.replace("${MATERIAL_ID}", material_name)

    # Optional direct overrides for simulation timing
    if start_year is not None:
        variant = replace_start_year(variant, start_year)
    if duration_years is not None:
        variant = replace_single_parameter(variant, "Duration", f"{duration_years}")

    return variant


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate Delphin material sensitivity variants."
    )
    parser.add_argument(
        "--template",
        default="Delphin_template_10y.d6p",
        help="Template Delphin project file relative to the working directory.",
    )
    parser.add_argument(
        "--base-dir",
        default=".",
        help="Project root containing template, materials/, and climate file. Default: current directory.",
    )
    parser.add_argument(
        "--variants-dir",
        default="variants",
        help="Output directory for generated .d6p variants.",
    )
    parser.add_argument(
        "--results-dir",
        default="results",
        help="Results directory placeholder / bookkeeping directory.",
    )
    parser.add_argument(
        "--job-list",
        default="job_list.txt",
        help="File listing one .d6p path per line for PBS array jobs.",
    )
    parser.add_argument(
        "--metadata",
        default="variant_metadata.tsv",
        help="TSV file describing generated variants.",
    )
    parser.add_argument(
        "--climate-file",
        default="KMI_Brussels_1987-2017_4CY.wac",
        help="Climate data file relative to base-dir.",
    )
    parser.add_argument(
        "--solar",
        nargs="+",
        type=float,
        default=[0.3, 0.5, 0.7],
        help="One or more solar absorption values to inject into the template.",
    )
    parser.add_argument(
        "--orientations",
        nargs="+",
        type=int,
        default=[0, 45, 90, 135, 180, 225, 270, 315],
        help="One or more orientation values (degrees) to inject into the template.",
    )
    parser.add_argument(
        "--wall-factors",
        nargs="+",
        type=float,
        default=[0.1, 0.2, 0.3, 0.5],
        help="One or more wall factors. re_value is computed from each wall factor.",
    )
    parser.add_argument(
        "--brick-types",
        nargs="+",
        default=["ZD", "ZA", "ZB", "ZF"],
        help="Brick type codes to include (e.g. ZA ZB ZD ZF). Filters materials by type code.",
    )
    parser.add_argument(
        "--start-year",
        type=int,
        default=2007,
        help="Override template StartYear. Default 2007 to avoid end-of-climate-range issues.",
    )
    parser.add_argument(
        "--duration-years",
        type=float,
        default=None,
        help="Optional override for the first Duration parameter with unit 'a'. Example: 6 or 5.99",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help=(
            "Skip jobs whose expected restart marker already exists under "
            "<variant_without_suffix>/var/restart.bin."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    base_dir = Path(args.base_dir).resolve()
    template_path = (base_dir / args.template).resolve()
    variants_dir = (base_dir / args.variants_dir).resolve()
    results_dir = (base_dir / args.results_dir).resolve()
    materials_dir = (base_dir / "Delphin_Brick").resolve()
    climate_file = (base_dir / args.climate_file).resolve()
    job_list_path = (base_dir / args.job_list).resolve()
    metadata_path = (base_dir / args.metadata).resolve()

    if not template_path.exists():
        raise FileNotFoundError(f"Template not found: {template_path}")
    if not materials_dir.exists():
        raise FileNotFoundError(f"materials directory not found: {materials_dir}")
    if not climate_file.exists():
        raise FileNotFoundError(f"Climate file not found: {climate_file}")

    template_text = read_text(template_path)
    materials = discover_materials_from_folder(materials_dir)
    if args.brick_types:
        materials = [m for m in materials if extract_brick_type(m) in args.brick_types]
        if not materials:
            raise RuntimeError(f"No materials matched brick types: {args.brick_types}")

    variants_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    materials_dir_for_delphin = materials_dir.as_posix()
    climate_file_for_delphin = climate_file.as_posix()

    queued_jobs: List[Path] = []
    metadata_rows: List[Tuple[str, str, float, int, float, int, str]] = []

    for wall_factor in args.wall_factors:
        re_value = (1.018 * wall_factor) / 0.3221376

        for orientation in args.orientations:
            for solar in args.solar:
                for material_name in materials:
                    variant_text = build_variant_text(
                        template_text=template_text,
                        materials_dir_for_delphin=materials_dir_for_delphin,
                        climate_file_for_delphin=climate_file_for_delphin,
                        re_value=re_value,
                        orientation_value=orientation,
                        solar_value=solar,
                        material_name=material_name,
                        start_year=args.start_year,
                        duration_years=args.duration_years,
                    )

                    safe_name = extract_brick_type(material_name)
                    variant_name = (
                        f"wf_{wall_factor:.3f}_re_{re_value:.6f}"
                        f"_ori_{orientation}_sol_{solar:.3f}_{safe_name}.d6p"
                    )
                    variant_path = variants_dir / variant_name
                    write_text(variant_path, variant_text)

                    restart_marker = variant_path.with_suffix("") / "var" / "restart.bin"
                    if args.skip_existing and restart_marker.exists():
                        continue

                    queued_jobs.append(variant_path)
                    metadata_rows.append(
                        (
                            variant_name,
                            material_name,
                            re_value,
                            orientation,
                            solar,
                            args.start_year,
                            climate_file_for_delphin,
                        )
                    )

    write_text(
        job_list_path,
        "\n".join(path.as_posix() for path in queued_jobs) + ("\n" if queued_jobs else ""),
    )

    with metadata_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(
            [
                "variant_file",
                "material_name",
                "re_value",
                "orientation",
                "solar",
                "start_year",
                "climate_file",
            ]
        )
        writer.writerows(metadata_rows)

    print(f"Brick types           : {args.brick_types}")
    print(f"Wall factors          : {args.wall_factors}")
    print(f"Orientations          : {args.orientations}")
    print(f"Solar values          : {args.solar}")
    print(f"Variants generated    : {len(metadata_rows)}")
    print(f"PBS job list          : {job_list_path}")
    print(f"Variant metadata      : {metadata_path}")
    print(f"Results directory     : {results_dir}")
    print(f"Start year   : {args.start_year}")
    print(f"Duration     : {args.duration_years}")


if __name__ == "__main__":
    main()
