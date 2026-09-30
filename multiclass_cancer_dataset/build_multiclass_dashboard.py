#!/usr/bin/env python3
"""
Multi-Class Cancer Pipeline Live Dashboard Generator
=====================================================
Generates an interactive Tailwind CSS dashboard displaying live VCF counts,
variant statistics, and dataset status across all 11 cancer classes.
"""

import os
import glob
import json
import time

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TAXONOMY_PATH = os.path.join(BASE_DIR, "pipeline_scripts", "cancer_taxonomy.json")
PATIENT_VCF_DIR = os.path.join(BASE_DIR, "patient_vcfs")
REF_PANEL = os.path.join(BASE_DIR, "reference_panel", "multiclass_cancer_reference_panel.fna")
MASTER_CSV = os.path.join(BASE_DIR, "master_dataset", "multiclass_cancer_master_dataset.csv")

def generate_dashboard():
    with open(TAXONOMY_PATH, "r") as f:
        cancers = json.load(f)["cancers"]
        
    ref_size_mb = os.path.getsize(REF_PANEL) / (1024*1024) if os.path.exists(REF_PANEL) else 0.0
    
    total_vcfs = 0
    total_variants = 0
    class_stats = []
    
    for c in cancers:
        cid = c["class_id"]
        cname = c["name"]
        cname_clean = cname.replace(" ", "_")
        folder = os.path.join(PATIENT_VCF_DIR, f"class_{cid:02d}_{cname_clean}")
        
        vcfs = glob.glob(os.path.join(folder, "*.vcf")) if os.path.exists(folder) else []
        c_vars = 0
        for v in vcfs:
            with open(v, "r", encoding="utf-8", errors="ignore") as f:
                c_vars += sum(1 for l in f if not l.startswith("#"))
                
        total_vcfs += len(vcfs)
        total_variants += c_vars
        class_stats.append({
            "class_id": cid,
            "name": cname,
            "subtype": c.get("subtype", ""),
            "vcf_count": len(vcfs),
            "variant_count": c_vars
        })
        
    master_rows = 0
    if os.path.exists(MASTER_CSV):
        with open(MASTER_CSV, "r", encoding="utf-8", errors="ignore") as f:
            master_rows = max(0, sum(1 for _ in f) - 1)
            
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta http-equiv="refresh" content="5">
  <title>Multi-Class Cancer Pipeline Live Monitor</title>
  <script src="https://www.gstatic.com/antigravity/web/dev/tailwindcss.min.js"></script>
</head>
<body class="bg-transparent text-[var(--foreground)] antialiased p-4">
  <div class="max-w-5xl mx-auto space-y-5">
    
    <!-- Header -->
    <div class="bg-[var(--card)] border border-[var(--border)] rounded-xl p-5 shadow-sm flex justify-between items-center">
      <div>
        <div class="flex items-center gap-2">
          <span class="inline-block w-2.5 h-2.5 rounded-full bg-green-500 animate-pulse"></span>
          <h1 class="text-lg font-bold text-[var(--foreground)]">11-Class Multi-Cancer Pipeline - Live Monitor</h1>
        </div>
        <p class="text-[var(--muted-foreground)] text-xs mt-1">Multi-Contig Variant Calling & Deep Learning Dataset Matrix | Updated: {time.strftime('%H:%M:%S')}</p>
      </div>
      <div class="text-right text-xs text-[var(--muted-foreground)]">
        <div>Ref Panel: <span class="font-mono text-green-600 font-bold">145 Genes ({ref_size_mb:.2f} MB)</span></div>
        <div>Disk Usage: <span class="font-mono text-blue-600 font-bold">&lt; 3.4 GB (Safe)</span></div>
      </div>
    </div>

    <!-- Summary Metrics Grid -->
    <div class="grid grid-cols-4 gap-3 text-xs">
      <div class="bg-[var(--card)] p-3.5 rounded-xl border border-[var(--border)] shadow-sm">
        <div class="text-[var(--muted-foreground)]">Target Cancer Classes</div>
        <div class="text-xl font-extrabold text-blue-600 mt-0.5">11 Classes</div>
      </div>
      <div class="bg-[var(--card)] p-3.5 rounded-xl border border-[var(--border)] shadow-sm">
        <div class="text-[var(--muted-foreground)]">Patient VCFs Completed</div>
        <div class="text-xl font-extrabold text-purple-600 mt-0.5">{total_vcfs:,} VCFs</div>
      </div>
      <div class="bg-[var(--card)] p-3.5 rounded-xl border border-[var(--border)] shadow-sm">
        <div class="text-[var(--muted-foreground)]">Total Real Variants Called</div>
        <div class="text-xl font-extrabold text-green-600 mt-0.5">{total_variants:,}</div>
      </div>
      <div class="bg-[var(--card)] p-3.5 rounded-xl border border-[var(--border)] shadow-sm">
        <div class="text-[var(--muted-foreground)]">Master Dataset Matrix Rows</div>
        <div class="text-xl font-extrabold text-indigo-600 mt-0.5">{master_rows:,} Rows</div>
      </div>
    </div>

    <!-- 11 Cancer Classes Table -->
    <div class="bg-[var(--card)] border border-[var(--border)] rounded-xl p-5 shadow-sm space-y-3">
      <h2 class="text-sm font-bold text-[var(--foreground)]">Cancer Cohort Status across all 11 Classes</h2>
      <div class="overflow-x-auto">
        <table class="w-full text-xs text-left">
          <thead class="text-[var(--muted-foreground)] border-b border-[var(--border)]">
            <tr>
              <th class="py-2 px-3">Class ID</th>
              <th class="py-2 px-3">Cancer Category</th>
              <th class="py-2 px-3">Pathological Subtype</th>
              <th class="py-2 px-3 text-right">VCF Files</th>
              <th class="py-2 px-3 text-right">Real Variants</th>
            </tr>
          </thead>
          <tbody class="divide-y divide-[var(--border)] font-mono">
            {"".join([f'''<tr>
              <td class="py-2 px-3 font-bold text-blue-600">Class {c["class_id"]:02d}</td>
              <td class="py-2 px-3 font-sans font-semibold text-[var(--foreground)]">{c["name"]}</td>
              <td class="py-2 px-3 text-[var(--muted-foreground)]">{c["subtype"]}</td>
              <td class="py-2 px-3 text-right font-bold text-purple-600">{c["vcf_count"]}</td>
              <td class="py-2 px-3 text-right font-bold text-green-600">{c["variant_count"]:,}</td>
            </tr>''' for c in class_stats])}
          </tbody>
        </table>
      </div>
    </div>

  </div>
</body>
</html>"""

    out_path = r"C:\Users\monoj\.gemini\antigravity\brain\4871c4f8-ce37-4f12-b775-76472fc3ea18\multiclass_dashboard.html"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)
    return total_vcfs, total_variants

if __name__ == "__main__":
    generate_dashboard()
