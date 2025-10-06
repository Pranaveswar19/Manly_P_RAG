#!/usr/bin/env python
import argparse
import os
import re
import shutil
import subprocess
import warnings
from pathlib import Path
from typing import List, Tuple, Optional

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"

PDF_INPUT_DIR = DATA_DIR / "pdfs"        # scraper output
OCR_OUTPUT_DIR = DATA_DIR / "ocr_pdfs"   # OCR’d PDFs
TXT_OUTPUT_DIR = DATA_DIR / "ocr_txt"    # extracted text

import fitz
try:
    fitz.TOOLS.mupdf_display_errors(False)
except Exception:
    pass

from ocrmypdf import ocr, exceptions as ocr_ex

def which_any(names: List[str]) -> Optional[str]:
    for n in names:
        p = shutil.which(n)
        if p:
            return p
    return None

def run_cmd(cmd: List[str]) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, check=False)


def get_version_string(cmd: str, args: List[str]) -> Optional[str]:
    try:
        out = subprocess.check_output([cmd, *args], stderr=subprocess.STDOUT, text=True)
        return out.strip().splitlines()[0]
    except Exception:
        return None


def print_env_summary():
    tools = [
        ("Ghostscript", ["gswin64c", "gswin32c", "gs"], ["-v"]),
        ("QPDF", ["qpdf"], ["--version"]),
        ("Tesseract", ["tesseract"], ["--version"]),
        ("pngquant (optional)", ["pngquant"], ["--version"]),
        ("jbig2 (optional)", ["jbig2", "jbig2enc"], ["--version"]),
        ("unpaper (optional)", ["unpaper"], ["--version"]),
    ]
    print("[ENV] Dependency summary:")
    for name, candidates, ver_args in tools:
        found = None
        for c in candidates:
            if shutil.which(c):
                v = get_version_string(c, ver_args) or "(version unknown)"
                found = f"{c}: {v}"
                break
        if found:
            print(f"  OK  {name}: {found}")
        else:
            print(f"  ... {name}: not found")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="OCR + text extraction for PDFs")
    parser.add_argument("-i", "--input", type=Path, default=Path("data"), help="Input directory containing PDFs (recursive)")
    parser.add_argument("-o", "--output", type=Path, default=None, help="Output directory for OCR'd PDFs; defaults to <input>_ocr")
    parser.add_argument("--text-dir", type=Path, default=None, help="Output directory for extracted .txt; defaults to <input>_txt")
    parser.add_argument("--combined-output", type=Path, default=Path("all_docs_text.txt"), help="Combined text output file ('' disables)")
    parser.add_argument("-l", "--lang", type=str, default="eng", help="OCR languages (e.g., 'eng', 'eng+deu'). Ensure traineddata installed.")
    parser.add_argument("--force-ocr", action="store_true", help="Force OCR on all pages (even if text is detected)")
    parser.add_argument("--no-skip-text", action="store_true", help="Do not skip pages with digital text (default skips)")
    parser.add_argument("--no-rotate", action="store_true", help="Disable automatic page rotation")
    parser.add_argument("--no-deskew", action="store_true", help="Disable deskew")
    parser.add_argument("--clean", choices=["auto", "on", "off"], default="auto", help="Use unpaper cleanup (auto uses if available)")
    parser.add_argument("--optimize", type=int, default=-1, help="Optimization level 0-3 (-1 auto: 3 if pngquant+jbig2, else 1)")
    parser.add_argument("-j", "--jobs", type=int, default=max(1, (os.cpu_count() or 2)), help="Parallel jobs for OCRmyPDF")
    parser.add_argument("--reprocess", action="store_true", help="Recreate OCR outputs even if target files exist")
    parser.add_argument("--fail-fast", action="store_true", help="Stop on first failure (default: continue)")
    parser.add_argument("--no-auto-repair", action="store_true", help="Disable auto-repair attempts for problematic PDFs")
    parser.add_argument("--no-preserve-indents", action="store_true", help="Strip leading spaces in extracted text (default preserves)")
    return parser.parse_args()

def configure_ocr_options(args: argparse.Namespace) -> dict:
    unpaper = which_any(["unpaper"])
    pngquant = which_any(["pngquant"])
    jbig2 = which_any(["jbig2", "jbig2enc"])

    if args.clean == "on":
        clean = True
    elif args.clean == "off":
        clean = False
    else:
        clean = bool(unpaper)

    if args.optimize >= 0:
        optimize = args.optimize
    else:
        optimize = 3 if (pngquant and jbig2) else 1

    opts = dict(
        language=args.lang,
        rotate_pages=not args.no_rotate,
        deskew=not args.no_deskew,
        skip_text=not args.no_skip_text,
        force_ocr=args.force_ocr,
        progress_bar=False,
        jobs=args.jobs,
        clean=clean,
        optimize=optimize,
    )

    print("[CFG] OCR options:")
    for k in ["language", "rotate_pages", "deskew", "skip_text", "force_ocr", "clean", "optimize", "jobs"]:
        print(f"  {k}: {opts[k]}")
    if clean and not unpaper:
        print("  note: clean=True requested but unpaper not found; OCRmyPDF may fail.")
    if opts["optimize"] >= 2 and not pngquant:
        print("  note: optimize>=2 requested but pngquant not found; OCRmyPDF may reduce to 1/0.")
    if opts["optimize"] >= 2 and not jbig2:
        print("  note: optimize>=2 requested without jbig2; OCRmyPDF may warn but continue.")

    return opts


def ensure_dirs(*dirs: Path):
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)


def list_pdfs(root: Path) -> List[Path]:
    if not root.exists():
        print(f"[WARN] Input directory does not exist: {root}")
        return []
    return [p for p in root.rglob("*.pdf") if p.is_file()]


def try_repair_with_qpdf(src: Path) -> Optional[Path]:
    if not which_any(["qpdf"]):
        return None
    repaired = src.with_suffix(".repaired.pdf")
    print(f"[REPAIR] qpdf rewriting: {src.name} -> {repaired.name}")
    proc = run_cmd(["qpdf", str(src), str(repaired)])
    if proc.returncode == 0 and repaired.exists() and repaired.stat().st_size > 0:
        return repaired
    repaired_lin = src.with_suffix(".repaired.linearized.pdf")
    proc2 = run_cmd(["qpdf", "--linearize", str(src), str(repaired_lin)])
    if proc2.returncode == 0 and repaired_lin.exists() and repaired_lin.stat().st_size > 0:
        return repaired_lin
    return None


def try_repair_with_ghostscript(src: Path) -> Optional[Path]:
    gs = which_any(["gswin64c", "gswin32c", "gs"])
    if not gs:
        return None
    repaired = src.with_suffix(".repaired.gs.pdf")
    print(f"[REPAIR] Ghostscript rebuilding: {src.name} -> {repaired.name}")
    cmd = [
        gs,
        "-o", str(repaired),
        "-sDEVICE=pdfwrite",
        "-dCompatibilityLevel=1.5",
        "-dPDFSETTINGS=/prepress",
        "-dDetectDuplicateImages=false",
        "-dColorImageDownsampleType=/None",
        "-dGrayImageDownsampleType=/None",
        "-dMonoImageDownsampleType=/Subsample",
        "-f", str(src),
    ]
    proc = run_cmd(cmd)
    if proc.returncode == 0 and repaired.exists() and repaired.stat().st_size > 0:
        return repaired
    return None


def ocrmypdf_file(src_pdf: Path, dst_pdf: Path, ocr_kw: dict, reprocess: bool, auto_repair: bool) -> bool:
    dst_pdf.parent.mkdir(parents=True, exist_ok=True)

    if dst_pdf.exists() and not reprocess:
        print(f"[SKIP] Exists: {dst_pdf}")
        return True

    if dst_pdf.exists():
        try:
            dst_pdf.unlink()
        except Exception:
            try:
                tmp = dst_pdf.with_suffix(dst_pdf.suffix + ".old")
                dst_pdf.rename(tmp)
            except Exception:
                pass

    def run_ocr(inp: Path, kw: dict, delete_after_success: Optional[Path] = None) -> bool:
        ocr(str(inp), str(dst_pdf), **kw)
        if delete_after_success and delete_after_success.exists():
            try:
                delete_after_success.unlink()
            except Exception:
                pass
        return True

    try:
        return run_ocr(src_pdf, ocr_kw)

    except ocr_ex.MissingDependencyError as e:
        msg = str(e).lower()
        print(f"[OCR] Missing dependency for {src_pdf.name}: {msg}")
        downgraded = False
        if "unpaper" in msg and ocr_kw.get("clean", False):
            ocr_kw = dict(ocr_kw, clean=False)
            print("[RETRY] clean=False (unpaper missing)")
            downgraded = True
        if ("pngquant" in msg or "jbig2" in msg) and ocr_kw.get("optimize", 0) > 0:
            new_opt = 1 if ocr_kw.get("optimize", 0) > 1 else 0
            ocr_kw = dict(ocr_kw, optimize=new_opt)
            print(f"[RETRY] optimize={new_opt} (pngquant/jbig2 missing)")
            downgraded = True
        if downgraded:
            try:
                return run_ocr(src_pdf, ocr_kw)
            except Exception as e2:
                print(f"[OCR] Retry failed {src_pdf.name}: {e2}")
        return False

    except (ocr_ex.SubprocessOutputError, Exception) as e:
        msg_raw = str(e)
        msg = msg_raw.lower()
        print(f"[OCR] Error {src_pdf.name}: {msg_raw}")

        if (("jbig2" in msg or "jbig2enc" in msg or "the system cannot find the file specified" in msg)
                and ocr_kw.get("optimize", 0) >= 2):
            safe_kw = dict(ocr_kw, optimize=1, clean=False)
            print("[RETRY] Downgrading optimize=1 (avoid jbig2 paths) and retrying...")
            try:
                return run_ocr(src_pdf, safe_kw)
            except Exception as e2:
                print(f"[OCR] Retry after downgrade failed {src_pdf.name}: {e2}")

        corruption_markers = [
            "image file is truncated",
            "corrupt jpeg",
            "invalid jpeg data reading from buffer",
            "invalid jpeg marker",
            "file is damaged",
            "xref",
            "error decoding stream data",
        ]
        if auto_repair and any(s in msg for s in corruption_markers):
            repaired = try_repair_with_qpdf(src_pdf) or try_repair_with_ghostscript(src_pdf)
            if repaired:
                print(f"[RETRY] Using repaired file: {repaired.name}")
                safe_kw = dict(ocr_kw, optimize=0, clean=False)
                try:
                    return run_ocr(repaired, safe_kw, delete_after_success=repaired)
                except Exception as e2:
                    print(f"[OCR] Retry with repaired file failed {src_pdf.name}: {e2}")

        return False


def clean_text_block(text: str, preserve_indents: bool) -> str:
    text = text.replace("\t", "    ")
    text = re.sub(r"-\n(?=\w)", "", text)
    lines = text.splitlines()
    if preserve_indents:
        lines = [re.sub(r"[ \t]+$", "", ln) for ln in lines]
        pretty = "\n".join(lines)
        pretty = re.sub(r"\n{4,}", "\n\n\n", pretty)
        return pretty
    else:
        text2 = "\n".join(ln.strip() for ln in lines)
        text2 = re.sub(r"\n{3,}", "\n\n", text2)
        return text2.strip()


def extract_text_reading_order(pdf_path: Path, preserve_indents: bool) -> str:
    parts: List[str] = []
    with fitz.open(pdf_path) as doc:
        for page_num, page in enumerate(doc, start=1):
            blocks: List[Tuple] = page.get_text("blocks")
            blocks_sorted = sorted(
                (b for b in blocks if isinstance(b[4], str) and b[4].strip("\n\r \t")),
                key=lambda b: (round(b[1], 1), round(b[0], 1)),
            )
            page_text_parts: List[str] = []
            prev_y1 = None
            for (x0, y0, x1, y1, text, *_rest) in blocks_sorted:
                t = clean_text_block(text, preserve_indents=preserve_indents)
                if not t.strip("\n\r "):
                    continue
                if prev_y1 is not None and (y0 - prev_y1) > 12:
                    page_text_parts.append("")
                page_text_parts.append(t)
                prev_y1 = y1
            page_text = "\n".join(page_text_parts).rstrip()
            if page_text:
                parts.append(f"=== PAGE {page_num} ===\n{page_text}")
    return "\n\n".join(parts).rstrip()


def write_text_outputs(txt: str, per_file_txt_path: Path, combined_handle):
    per_file_txt_path.parent.mkdir(parents=True, exist_ok=True)
    per_file_txt_path.write_text(txt, encoding="utf-8")
    if combined_handle is not None:
        combined_handle.write(f"--- {per_file_txt_path.stem}.pdf ---\n")
        combined_handle.write(txt)
        combined_handle.write("\n\n")


def main():
    args = parse_args()
    print_env_summary()

    input_dir: Path = PDF_INPUT_DIR
    BASE_DIR = Path(__file__).resolve().parents[1]
    DATA_DIR = BASE_DIR / "data"

    ocr_dir: Path = args.output or OCR_OUTPUT_DIR
    txt_dir: Path = args.text_dir or TXT_OUTPUT_DIR
    ensure_dirs(ocr_dir, txt_dir)

    ocr_opts = configure_ocr_options(args)
    pdfs = list_pdfs(input_dir)
    if not pdfs:
        print(f"[INFO] No PDFs found under {input_dir}")
        return

    combined_path = args.combined_output if str(args.combined_output) else None
    combined_fh = None
    processed = 0

    try:
        if combined_path:
            combined_fh = open(combined_path, "w", encoding="utf-8")

        for src in pdfs:
            rel = src.relative_to(input_dir)
            dst_pdf = (ocr_dir / rel).with_suffix(".pdf") 
            print(f"[OCR] {src} -> {dst_pdf}")
            ok = ocrmypdf_file(
                src, dst_pdf, dict(ocr_opts),
                reprocess=args.reprocess,
                auto_repair=not args.no_auto_repair
            )
            if not ok:
                print(f"[SKIP] OCR failed: {src}")
                if args.fail_fast:
                    break
                continue

            print(f"[TEXT] Extract: {dst_pdf}")
            txt = extract_text_reading_order(dst_pdf, preserve_indents=not args.no_preserve_indents)
            per_file_txt = (txt_dir / rel).with_suffix(".txt")
            write_text_outputs(txt, per_file_txt, combined_fh)
            processed += 1

    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user (Ctrl+C). Partial results are kept.")
    finally:
        if combined_fh:
            combined_fh.close()

    print(f"[DONE] Processed {processed} PDFs")
    if combined_path:
        print(f"[OUT] Combined text: {Path(combined_path).resolve()}")
    print(f"[OUT] OCR PDFs dir: {ocr_dir.resolve()}")
    print(f"[OUT] Text dir: {txt_dir.resolve()}")


if __name__ == "__main__":
    warnings.filterwarnings("ignore", message=r"image file is truncated.*")
    main()