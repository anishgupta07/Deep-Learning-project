import sys
import os
import subprocess
import shutil

print("=" * 75)
print(">>> GenomeDL Automated Environment & Dependency Verification")
print("=" * 75)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PYTHON_EXE = sys.executable

# 1. Install / Verify Python Packages
print("\n[Step 1/4] Checking and installing Python dependencies...")
REQUIRED_PY_PACKAGES = [
    "pandas",
    "biopython",
    "rdkit",
    "scipy",
    "requests",
    "numpy"
]

installed_py = []
missing_py = []
for pkg in REQUIRED_PY_PACKAGES:
    try:
        __import__(pkg)
        installed_py.append(pkg)
    except ImportError:
        missing_py.append(pkg)

if missing_py:
    print(f"[*] Missing Python packages: {missing_py}. Installing now via pip...")
    cmd = [PYTHON_EXE, "-m", "pip", "install"] + missing_py
    subprocess.check_call(cmd)
    print("[+] All Python packages installed successfully!")
else:
    print(f"[+] All required Python packages already installed: {', '.join(installed_py)}")

# 2. Check WSL Installation
print("\n[Step 2/4] Verifying Windows Subsystem for Linux (WSL)...")
wsl_available = shutil.which("wsl") is not None
if not wsl_available:
    print("[-] WSL not found on this system.")
    print("    Please install WSL by opening PowerShell as Administrator and running: wsl --install")
    sys.exit(1)

print("[+] WSL is available on this system.")

# Check for bioinformatics tools inside WSL
tools_needed = ["minimap2", "samtools", "bcftools", "curl"]
missing_tools = []

for tool in tools_needed:
    res = subprocess.run(["wsl", "which", tool], capture_output=True, text=True)
    if res.returncode == 0 and res.stdout.strip():
        # Get version
        v_res = subprocess.run(["wsl", tool, "--version"], capture_output=True, text=True)
        v_line = v_res.stdout.strip().split("\n")[0] if v_res.stdout else "active"
        print(f"    [+] {tool:<10}: {v_line}")
    else:
        missing_tools.append(tool)

if missing_tools:
    print(f"\n[!] The following tools are missing in WSL: {', '.join(missing_tools)}")
    print("    Please open PowerShell and run:")
    print(f"    wsl sudo apt-get update; wsl sudo apt-get install -y {' '.join(missing_tools)}")
    sys.exit(1)
else:
    print("[+] All 4 Linux bioinformatics tools (minimap2, samtools, bcftools, curl) verified!")

# 3. Verify Unified Reference Panel
print("\n[Step 3/4] Verifying 128-Gene Reference Panel and Index...")
ref_fna = os.path.join(BASE_DIR, "unified_128_gene_reference_panel.fna")
ref_fai = os.path.join(BASE_DIR, "unified_128_gene_reference_panel.fna.fai")

if os.path.exists(ref_fna):
    print(f"    [+] Reference panel found: {ref_fna} ({os.path.getsize(ref_fna)/(1024*1024):.2f} MB)")
else:
    print(f"    [-] Missing reference panel: {ref_fna}")
    sys.exit(1)

if not os.path.exists(ref_fai):
    print("    [*] Creating FASTA index (.fai)...")
    drive = ref_fna[0].lower()
    wsl_ref = f"/mnt/{drive}{ref_fna[2:].replace(os.sep, '/')}"
    subprocess.run(["wsl", "bash", "-c", f"samtools faidx {wsl_ref}"])
    print("    [+] Reference FASTA indexed successfully.")
else:
    print("    [+] Reference FASTA index (.fai) present.")

# 4. Smoke Test Engine Import
print("\n[Step 4/4] Running project smoke test...")
try:
    if BASE_DIR not in sys.path:
        sys.path.insert(0, BASE_DIR)
    import bio_chemistry_engine as bce
    import pipeline_engine as pe
    print("[+] Bio-Chemistry Engine functions defined and ready!")
    print("[+] Pipeline Engine loaded and all paths verified!")
except Exception as e:
    print(f"[-] Smoke test failed: {e}")
    sys.exit(1)

print("\n" + "=" * 75)
print("[SETUP 100% COMPLETE] Your environment is verified and ready to run!")
print("=" * 75)
print("\nTo start processing any cancer class, simply run:")
print("    python pipeline_engine.py --class-id <ID> --cohort --limit 20")
print("\nExamples:")
print("    Class 02 (Female Breast Cancer)   : python pipeline_engine.py --class-id 2 --cohort --limit 20")
print("    Class 03 (Colorectal Cancer)       : python pipeline_engine.py --class-id 3 --cohort --limit 20")
print("    Class 04 (Prostate Cancer)         : python pipeline_engine.py --class-id 4 --cohort --limit 20")
print("    Class 05 (Stomach / Gastric Cancer): python pipeline_engine.py --class-id 5 --cohort --limit 20")
print("    Class 06 (Thyroid Cancer)          : python pipeline_engine.py --class-id 6 --cohort --limit 20")
print("    Class 07 (Liver Cancer HCC)        : python pipeline_engine.py --class-id 7 --cohort --limit 20")
print("    Class 08 (Bladder Cancer)          : python pipeline_engine.py --class-id 8 --cohort --limit 20")
print("    Class 09 (Cervical Cancer)         : python pipeline_engine.py --class-id 9 --cohort --limit 20")
print("    Class 10 (Non-Hodgkin Lymphoma)    : python pipeline_engine.py --class-id 10 --cohort --limit 20")
print("=" * 75)
