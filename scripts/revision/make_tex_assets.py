#!/usr/bin/env python
"""
Turn the revision result files into LaTeX tables + number macros
(paper_revision/generated/*.tex). Re-run after new seeds finish; every number in
the revised manuscript that comes from an experiment is produced here.
"""
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import softmax

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.revision.common import ROOT, OUT, CLASSES, PAPER_RUNS, load_dump, fit_temperature  # noqa: E402

G = ROOT / "paper_revision" / "generated"
G.mkdir(parents=True, exist_ok=True)
S42 = OUT / "seed42"
MS = OUT / "multiseed"
OFF = OUT / "official_test"
INT = OUT / "integrity"
GC = OUT / "gradcam"
macros = {}
# a hyphen directly before a number (not part of '--', a word or a decimal) is a minus sign;
# \ensuremath{-} typesets a true minus both in running text and inside math
_NEG = re.compile(r"(?<![\w.\-])-(?=\.?\d)")


def tminus(s):
    return _NEG.sub(lambda _: "\\ensuremath{-}", str(s))


def m(name, value, nd=3):
    if isinstance(value, (float, np.floating)):
        value = f"{value:.{nd}f}"
    macros[name] = tminus(value)


def f3(x, nd=3):
    return "--" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x:.{nd}f}"


def fp(p):
    """p-value for running text, used as $p=\\Macro$ (math mode): never prints 0.000."""
    if p is None or not np.isfinite(p):
        return "--"
    if p >= 0.001:
        return f"{p:.3f}"
    if p >= 1e-4:
        return f"{p:.4f}"
    mant, exp = f"{p:.0e}".split("e")
    return f"{mant}\\times10^{{{int(exp)}}}"


def ptab(p, label=False):
    """Compact p-value for table cells; label=True gives '$p=0.03$' / '$p<0.001$'."""
    if p is None or not np.isfinite(p):
        return "--"
    s = f"{p:.2f}" if p >= 0.01 else (f"{p:.3f}" if p >= 0.001 else "<0.001")
    if label:
        return f"$p{'' if s.startswith('<') else '='}{s}$"
    return f"${s}$" if s.startswith("<") else s


def write(name, body):
    (G / f"{name}.tex").write_text(body, encoding="utf-8")


def tabular(caption, label, header, rows, colspec, star=False, note=None, size="\\footnotesize"):
    env = "table*" if star else "table"
    lines = [f"\\begin{{{env}}}[{'!t' if star else '!htbp'}]", "\\centering", f"\\caption{{{caption}}}", f"\\label{{{label}}}", size,
             "\\setlength{\\tabcolsep}{3pt}", f"\\begin{{tabular}}{{{colspec}}}", "\\toprule",
             " & ".join(header) + " \\\\", "\\midrule"]
    for r in rows:
        lines.append("\\midrule" if r == "MIDRULE" else " & ".join(tminus(c) for c in r) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    if note:
        lines.append(f"\\par\\vspace{{2pt}}\\parbox{{\\linewidth}}{{\\scriptsize {note}}}")
    lines.append(f"\\end{{{env}}}")
    return "\n".join(lines) + "\n"


# ------------------------------------------------------------------ seed 42 point + CI
def seed42():
    a = pd.read_csv(S42 / "A_point_metrics_val_test.csv")
    c = pd.read_csv(S42 / "C_bootstrap_ci_test.csv")
    summ = json.loads((S42 / "summary.json").read_text())

    def ci(model, metric):
        r = c[(c.model == model) & (c.metric == metric)].iloc[0]
        return r.point, r.ci_low, r.ci_high, r.get("bootstrap_p_two_sided", np.nan)

    key = {"resnet50_E1": "One", "resnet50_E0": "Zero", "effnetb3_E1": "Eff"}
    for mk, tag in key.items():
        for split in ["val", "test"]:
            r = a[(a.model == mk) & (a.split == split)].iloc[0]
            sp = "Val" if split == "val" else "Test"
            m(f"E{tag}{sp}Acc", r.accuracy, 4)
            m(f"E{tag}{sp}BalAcc", r.balanced_accuracy, 4)
            m(f"E{tag}{sp}MacroFone", r.macro_f1, 4)
            m(f"E{tag}{sp}AUC", r.roc_auc_macro_ovr, 4)
            m(f"E{tag}{sp}ECEbefore", r.ECE_before)
            m(f"E{tag}{sp}ECEafter", r.ECE_afterT)
            m(f"E{tag}{sp}NLLbefore", r.NLL_before)
            m(f"E{tag}{sp}NLLafter", r.NLL_afterT)
        m(f"E{tag}T", summ["temperatures"][mk], 2)
    # E1 vs E0 differences
    rows = []
    for metric, lab in [("accuracy", "Accuracy"), ("balanced_accuracy", "Balanced accuracy"), ("macro_f1", "Macro-F1"),
                        ("roc_auc_macro_ovr", "ROC-AUC (macro, OvR)"), ("ECE_afterT", "ECE after $T$"),
                        ("NLL_afterT", "NLL after $T$"), ("AUROC_MSP_vs_deterr", "Error AUROC (MSP)"),
                        ("AURC_MSP", "AURC (MSP)"), ("sel_coverage", "Coverage (85\\% val.\\ target)"),
                        ("sel_accepted_acc", "Accepted-set accuracy")]:
        e0, e1 = ci("resnet50_E0", metric), ci("resnet50_E1", metric)
        d = ci("E1_minus_E0", metric)
        rows.append([lab, f"{e0[0]:.3f} [{e0[1]:.3f}, {e0[2]:.3f}]", f"{e1[0]:.3f} [{e1[1]:.3f}, {e1[2]:.3f}]",
                     f"{d[0]:+.3f} [{d[1]:+.3f}, {d[2]:+.3f}]"])
    write("tab_seed42_ci", tabular(
        "Seed-42 models (as submitted) on the internal test split with 95\\% lesion-cluster bootstrap intervals "
        "(2,000 resamples of 1,399 lesion groups). The last column is the paired difference.",
        "tab:seed42ci", ["Metric", "E0 (plain CE)", "E1 (weighted CE)", "E1 $-$ E0"], rows, "lccc", star=True))
    for metric, nm in [("balanced_accuracy", "BalAcc"), ("accuracy", "Acc"), ("macro_f1", "MacroFone"),
                       ("AURC_MSP", "AURC"), ("NLL_afterT", "NLLafter"), ("AUROC_MSP_vs_deterr", "AUROCmsp")]:
        d = ci("E1_minus_E0", metric)
        m(f"DiffEoneEzero{nm}", f"{d[0]:+.3f}")
        m(f"DiffEoneEzero{nm}CI", f"[{d[1]:+.3f}, {d[2]:+.3f}]")
        m(f"DiffEoneEzero{nm}P", fp(d[3]))
    for metric, nm in [("ECE_afterT", "ECEafter"), ("NLL_afterT", "NLLafter"), ("AUROC_MSP_vs_deterr", "AUROCmsp"),
                       ("AURC_MSP", "AURC"), ("AUROC_MCentropy_vs_MCerrors", "AUROCmcPaper"),
                       ("balanced_accuracy", "BalAcc"), ("macro_f1", "MacroFone"), ("accuracy", "Acc"),
                       ("roc_auc_macro_ovr", "AUC")]:
        p, lo, hi, _ = ci("resnet50_E1", metric)
        m(f"EOne{nm}CI", f"[{lo:.3f}, {hi:.3f}]")
    # model selection on validation
    vs = summ["validation_selection"]
    m("ValFoneResnet", vs["resnet50_val_macro_f1"])
    m("ValFoneEff", vs["effnet_val_macro_f1"])
    m("ValBalResnet", vs["resnet50_val_bal_acc"])
    m("ValBalEff", vs["effnet_val_bal_acc"])

    # per-class table (E1 test)
    pc = pd.read_csv(S42 / "A_per_class_metrics.csv")
    rows = []
    for cl in CLASSES:
        r1 = pc[(pc.model == "resnet50_E1") & (pc.split == "test") & (pc["class"] == cl)].iloc[0]
        r0 = pc[(pc.model == "resnet50_E0") & (pc.split == "test") & (pc["class"] == cl)].iloc[0]
        rows.append([cl, str(int(r1.support)), f3(r0.recall), f3(r1.precision), f3(r1.recall), f3(r1.f1)])
    write("tab_perclass", tabular("Per-class test results (seed 42). E0 recall is shown for comparison with E1.",
                                  "tab:perclass", ["Class", "$n$", "E0 rec.", "E1 prec.", "E1 rec.", "E1 F1"],
                                  rows, "lrrrrr"))

    # calibration variants
    d = pd.read_csv(S42 / "D_calibration_variants_test.csv")
    names = {"uncalibrated": "none", "temperature": "temperature", "vector_scaling": "vector scaling",
             "prior_correction": "prior correction", "prior_correction+T": "prior corr.\\,+\\,$T$"}
    mod = {"resnet50_E1": "ResNet-50 E1", "resnet50_E0": "ResNet-50 E0", "effnetb3_E1": "EffNet-B3 E1"}
    rows = []
    for mk in ["resnet50_E1", "resnet50_E0", "effnetb3_E1"]:
        for _, r in d[d.model == mk].iterrows():
            rows.append([mod[mk] if r.variant == "uncalibrated" else "", names[r.variant], f3(r.ECE), f3(r.ECE_adaptive),
                         f3(r.ECE_classwise), f3(r.Brier), f3(r.NLL), f3(r.balanced_accuracy)])
        rows.append("MIDRULE")
    rows = rows[:-1]
    write("tab_calibration", tabular(
        "Calibration on the internal test split (seed 42). All maps are fitted on validation only. ECE: 15 equal-width "
        "bins; aECE: 15 equal-mass bins; cwECE: class-wise ECE. Vector scaling and prior correction change the "
        "decision rule, hence the last column.",
        "tab:calibration", ["Model", "Post-hoc map", "ECE", "aECE", "cwECE", "Brier", "NLL", "Bal.\\,acc."],
        rows, "llcccccc", star=True))
    r = d[(d.model == "resnet50_E1")].set_index("variant")
    m("VSECE", r.loc["vector_scaling", "ECE"])
    m("VScwECE", r.loc["vector_scaling", "ECE_classwise"])
    m("VSNLL", r.loc["vector_scaling", "NLL"])
    m("VSBal", r.loc["vector_scaling", "balanced_accuracy"])
    m("TcwECE", r.loc["temperature", "ECE_classwise"])
    m("UncwECE", r.loc["uncalibrated", "ECE_classwise"])
    m("TaECE", r.loc["temperature", "ECE_adaptive"])
    m("TBrier", r.loc["temperature", "Brier"])
    m("UnBrier", r.loc["uncalibrated", "Brier"])
    m("PCTECE", r.loc["prior_correction+T", "ECE"])
    m("PCTNLL", r.loc["prior_correction+T", "NLL"])
    m("PCTBal", r.loc["prior_correction+T", "balanced_accuracy"])
    m("PCTAcc", r.loc["prior_correction+T", "accuracy"])
    m("TafterPC", summ["T_after_prior_correction"]["resnet50_E1"], 2)

    # uncertainty scores
    e = pd.read_csv(S42 / "E_uncertainty_scores_test.csv")
    dd = pd.read_csv(S42 / "E_auroc_diff_vs_MSP_resnet50_E1.csv").set_index("score")
    lab = {"MSP_T": "MSP ($1-\\max_k p_k$), calibrated", "MSP_raw": "MSP, uncalibrated",
           "softmax_entropy_T": "Softmax entropy, calibrated", "MC_entropy_T": "MC predictive entropy (paper)",
           "MC_expected_entropy_T": "MC expected entropy (aleatoric)", "MC_mutual_info_T": "MC mutual information (epistemic)",
           "MC_variance_T": "MC predictive variance", "MC_MSP_T": "MC mean-probability MSP",
           "softmax_entropy_raw": "Softmax entropy, uncalibrated", "MC_entropy_raw": "MC predictive entropy, uncalibrated"}
    rows = []
    for sc in lab:
        r = e[(e.model == "resnet50_E1") & (e.score == sc)].iloc[0]
        dl = dd.loc[sc]
        rows.append([lab[sc], f3(r.AUROC_det_errors), "--" if sc == "MSP_T" else f"{dl.diff_vs_MSP_T:+.3f} [{dl.ci_low:+.3f}, {dl.ci_high:+.3f}]",
                     f3(r["pAUROC_fpr0.2_det_errors"]), f3(r.AURC_det)])
    tta_rows = OUT / "tta" / "tta_table_rows_resnet50_E1.csv"
    if tta_rows.exists():
        for _, r in pd.read_csv(tta_rows).iterrows():
            nm = {"TTA_entropy": "TTA predictive entropy", "TTA_mutual_info": "TTA mutual information"}[r.score]
            rows.append([nm, f3(r.AUROC), f"{r.diff_vs_MSP_T:+.3f} [{r.ci_low:+.3f}, {r.ci_high:+.3f}]",
                         f3(r.pAUROC), f3(r.AURC)])
    write("tab_uncertainty", tabular(
        "Error detection on the internal test split (ResNet-50 E1, seed 42). Target: errors of the deployed "
        "(deterministic) classifier. $\\Delta$: paired lesion-cluster bootstrap difference to calibrated MSP. "
        "pAUROC: standardised partial AUROC for FPR\\,$\\le$\\,0.2. AURC: area under the risk--coverage curve "
        "(random rejection = " + f3(float(pd.read_csv(S42 / 'F_aurc_val_test.csv').query("model=='resnet50_E1' and split=='test'").iloc[0]['overall_risk(random-rejection AURC)'])) + ").",
        "tab:uncertainty", ["Score", "AUROC", "$\\Delta$ vs MSP [95\\% CI]", "pAUROC", "AURC"], rows, "lcccc", star=True))
    for sc, nm in [("MSP_T", "MSP"), ("MC_entropy_T", "MCent"), ("MC_mutual_info_T", "MI"), ("softmax_entropy_T", "SMent"),
                   ("MC_expected_entropy_T", "ExpEnt")]:
        r = e[(e.model == "resnet50_E1") & (e.score == sc)].iloc[0]
        m(f"AUROC{nm}", r.AUROC_det_errors)
        m(f"AURC{nm}", r.AURC_det)
        if sc != "MSP_T":
            dl = dd.loc[sc]
            m(f"Diff{nm}", f"{dl.diff_vs_MSP_T:+.3f}")
            m(f"Diff{nm}CI", f"[{dl.ci_low:+.3f}, {dl.ci_high:+.3f}]")
    r = e[(e.model == "resnet50_E1") & (e.score == "MC_entropy_T")].iloc[0]
    m("AUROCMCentPaper", r.AUROC_own_errors)
    r = e[(e.model == "resnet50_E1") & (e.score == "MC_entropy_raw")].iloc[0]
    m("AUROCMCentRaw", r.AUROC_det_errors)
    m("AUROCMCentRawOwn", r.AUROC_own_errors)
    dl = dd.loc["MC_entropy_raw"]
    m("DiffMCentRaw", f"{dl.diff_vs_MSP_T:+.3f}")
    m("DiffMCentRawCI", f"[{dl.ci_low:+.3f}, {dl.ci_high:+.3f}]")
    cor = pd.read_csv(S42 / "E_score_correlations_test.csv").set_index("model")
    m("RhoMSPMCent", cor.loc["resnet50_E1", "spearman_MSP_vs_MCentropy"], 2)
    m("RhoMSPMI", cor.loc["resnet50_E1", "spearman_MSP_vs_MI"], 2)
    m("FracMCchanged", 100 * cor.loc["resnet50_E1", "frac_pred_changed_by_MC"], 1)
    cb = pd.read_csv(S42 / "E_combined_scores_test.csv")
    r = cb[(cb.model == "resnet50_E1") & (cb.features == "MSP_T+MC_entropy_T+MC_mutual_info_T")].iloc[0]
    m("AUROCcombined", r.test_AUROC)
    m("AURCcombined", r.test_AURC)

    # selective
    t = pd.read_csv(S42 / "F_threshold_transfer_val_to_test.csv")
    rows = []
    for _, r in t[(t.model == "resnet50_E1") & (t.score.isin(["MSP_T", "MSP_raw", "MC_entropy_T"]))].iterrows():
        if r.target_val_coverage not in (0.7, 0.8, 0.85, 0.9, 0.95):
            continue
        sc = {"MSP_T": "MSP (cal.)", "MSP_raw": "MSP (raw)", "MC_entropy_T": "MC entropy"}[r.score]
        rows.append([sc, f"{r.target_val_coverage:.2f}", f"{r.threshold:.3f}", f3(r.val_coverage), f3(r.test_coverage),
                     f3(r.test_selective_acc), f3(r.test_rejected_acc), f3(r.MEL_sensitivity_accepted),
                     str(int(r.MEL_cases_rejected))])
    write("tab_selective", tabular(
        "Validation-locked selective prediction (ResNet-50 E1, seed 42). Thresholds are chosen on validation for a "
        "target validation coverage and applied unchanged to test. Rej.\\ acc.: accuracy among rejected test cases; "
        "MEL sens.: melanoma sensitivity among accepted cases (all cases: " + f3(float(t.iloc[0].MEL_sensitivity_all)) + ").",
        "tab:selective", ["Score", "Target", "$\\tau$", "Val.\\ cov.", "Test cov.", "Sel.\\ acc.", "Rej.\\ acc.",
                          "MEL sens.", "MEL rej."], rows, "lcccccccc", star=True))
    r = t[(t.model == "resnet50_E1") & (t.score == "MSP_T") & (t.target_val_coverage == 0.85)].iloc[0]
    m("SelTau", r.threshold)
    m("SelValCov", r.val_coverage, 4)
    m("SelTestCov", r.test_coverage, 4)
    m("SelTestAcc", r.test_selective_acc, 4)
    m("SelNacc", int(r.test_n_accepted))
    m("SelNrej", int(r.test_n_rejected))
    m("SelRejAcc", r.test_rejected_acc)
    m("SelNcorrRej", int(r.test_n_correct_among_rejected))
    m("SelMELall", r.MEL_sensitivity_all)
    m("SelMELacc", r.MEL_sensitivity_accepted)
    m("SelMELrej", int(r.MEL_cases_rejected))
    m("SelMELcov", r.MEL_coverage)
    m("SelBalAccAcc", r.balanced_acc_accepted)
    au = pd.read_csv(S42 / "F_aurc_val_test.csv")
    r = au[(au.model == "resnet50_E1") & (au.split == "val") & (au.score == "MSP_T")].iloc[0]
    m("AURCvalMSP", r.AURC)
    m("RiskVal", r["overall_risk(random-rejection AURC)"])
    r = au[(au.model == "resnet50_E1") & (au.split == "test") & (au.score == "MSP_T")].iloc[0]
    m("RiskTest", r["overall_risk(random-rejection AURC)"])
    m("EAURCtestMSP", r.E_AURC)
    # class-wise selective
    cw = pd.read_csv(S42 / "F_classwise_selective_resnet50_E1.csv")
    rows = [[r["class"], str(int(r.n_test)), f3(r.coverage), str(int(r.n_rejected)), f3(r.recall_all), f3(r.recall_accepted)]
            for _, r in cw.iterrows()]
    write("tab_classwise_selective", tabular(
        "Class-wise behaviour of the validation-locked rule ($\\tau=" + macros["SelTau"] + "$) on the internal test split.",
        "tab:classwiseselective", ["Class", "$n$", "Coverage", "Rejected", "Recall (all)", "Recall (accepted)"],
        rows, "lrrrrr"))
    for _, r in cw.iterrows():
        m(f"Cov{r['class']}", r.coverage)
    # SGR
    sg = pd.read_csv(S42 / "F_SGR_risk_control.csv")
    rows = []
    for _, r in sg.iterrows():
        rows.append([mod[r.model], f"{r.target_risk:.2f}", f3(r.threshold), f3(r.risk_bound), f3(r.val_coverage),
                     f3(r.test_coverage), f3(r.test_risk)])
    write("tab_sgr_full", tabular(
        "Risk-controlled thresholds (SGR, Geifman \\& El-Yaniv) fitted on validation with $\\delta=0.05$ and applied to test.",
        "tab:sgrfull", ["Model", "$r^*$", "$\\theta$", "Bound", "Val.\\ cov.", "Test cov.", "Test risk"], rows, "lcccccc"))
    rows_e1 = [r[1:] for r, (_, s_) in zip(rows, sg.iterrows()) if s_.model == "resnet50_E1"]
    write("tab_sgr", tabular(
        "Risk-controlled thresholds (SGR) for ResNet-50 E1 (seed 42), fitted on validation with $\\delta=0.05$ and "
        "applied to test; the bound holds with probability 0.95 (all models: Supplementary~S5).",
        "tab:sgr", ["$r^*$", "$\\theta$", "Bound", "Val.\\ cov.", "Test cov.", "Test risk"], rows_e1, "cccccc"))
    for _, r in sg[sg.model == "resnet50_E1"].iterrows():
        k = {0.10: "Ten", 0.15: "Fifteen", 0.20: "Twenty", 0.25: "TwentyFive"}[round(r.target_risk, 2)]
        m(f"SGR{k}Cov", r.test_coverage)
        m(f"SGR{k}Risk", r.test_risk)
        m(f"SGR{k}ValCov", r.val_coverage)
    # conformal
    cf = pd.read_csv(S42 / "G_conformal_test.csv")
    rows = []
    ccw = pd.read_csv(S42 / "G_conformal_classwise_test.csv")
    for _, r in cf[cf.model == "resnet50_E1"].iterrows():
        q = ccw[(ccw.model == "resnet50_E1") & (ccw.method == r.method) & (np.isclose(ccw.alpha, r.alpha))]
        worst = f" ({q.loc[q.coverage.idxmin(), 'class']})" if len(q) else ""
        rows.append([{"LAC": "LAC", "APS": "APS", "classwise_LAC": "Class-cond.\\ LAC"}[r.method], f"{1 - r.alpha:.2f}",
                     f3(r.coverage), f3(r.avg_set_size), f3(r.frac_singleton), f3(r.acc_on_singletons),
                     f3(r.MEL_coverage), f3(r.min_class_coverage) + worst])
    write("tab_conformal", tabular(
        "Split-conformal prediction sets (ResNet-50 E1, seed 42; calibration set = validation, evaluation = internal test).",
        "tab:conformal", ["Method", "$1-\\alpha$", "Coverage", "Set size", "Singleton", "Acc.\\ singl.", "MEL cov.",
                          "Min.\\ class cov."], rows, "lccccccc", star=True))
    r = cf[(cf.model == "resnet50_E1") & (cf.method == "LAC") & (cf.alpha == 0.1)].iloc[0]
    m("ConfLACcov", r.coverage)
    m("ConfLACsize", r.avg_set_size, 2)
    m("ConfLACmin", r.min_class_coverage)
    m("ConfLACsingle", r.frac_singleton)
    m("ConfLACsingleAcc", r.acc_on_singletons)
    r = cf[(cf.model == "resnet50_E1") & (cf.method == "classwise_LAC") & (cf.alpha == 0.1)].iloc[0]
    m("ConfCWcov", r.coverage)
    m("ConfCWsize", r.avg_set_size, 2)
    m("ConfCWmin", r.min_class_coverage)
    cc = pd.read_csv(S42 / "G_conformal_classwise_test.csv")
    q = cc[(cc.model == "resnet50_E1") & (cc.alpha == 0.1) & (cc.method == "LAC")].set_index("class")
    m("ConfLACscc", q.loc["SCC", "coverage"])
    m("ConfLACak", q.loc["AK", "coverage"])


# ------------------------------------------------------------------ multi-seed
def multiseed():
    if not (MS / "summary_mean_sd.csv").exists():
        return
    s = pd.read_csv(MS / "summary_mean_sd.csv")
    p = pd.read_csv(MS / "paired_differences.csv")
    prot = s[s.protocol == "paper"]

    def ms(model, metric):
        r = prot[(prot.model == model) & (prot.metric == metric)]
        if not len(r):
            return "--", 0
        r = r.iloc[0]
        sd = "" if not np.isfinite(r.sd) else f" $\\pm$ {r.sd:.3f}"
        return f"{r['mean']:.3f}{sd}", int(r.n_seeds)

    rows = []
    mets = [("test_accuracy", "Accuracy"), ("test_balanced_accuracy", "Balanced accuracy"), ("test_macro_f1", "Macro-F1"),
            ("test_roc_auc_macro_ovr", "ROC-AUC"), ("test_ECE_before", "ECE (uncalibrated)"), ("test_ECE_afterT", "ECE ($T$)"),
            ("test_NLL_afterT", "NLL ($T$)"), ("test_Brier_afterT", "Brier ($T$)"), ("T", "Temperature $T$"),
            ("test_AUROC_MSP", "Error AUROC, MSP"), ("test_AUROC_MC_entropy_deterr", "Error AUROC, MC entropy"),
            ("test_AURC_MSP", "AURC, MSP"), ("test_sel_coverage", "Coverage at val.\\ 85\\%"),
            ("test_sel_acc", "Selective accuracy"), ("off_bal_acc_known", "Official test bal.\\ acc.$^\\dagger$"),
            ("off_ECE_afterT", "Official test ECE ($T$)$^\\dagger$"), ("off_OOD_AUROC_MSP", "UNK detection AUROC (MSP)$^\\dagger$")]
    nseeds = {}
    for met, lab in mets:
        cells = []
        for mk in ["resnet50_E0", "resnet50_E1", "effnetb3_E1"]:
            v, n = ms(mk, met)
            nseeds[mk] = max(nseeds.get(mk, 0), n)
            cells.append(v)
        q = p[(p.comparison == "resnet50_E1 - resnet50_E0") & (p.metric == met)]
        if len(q) and q.iloc[0].n_paired_seeds > 1:
            q = q.iloc[0]
            cells.append(f"{q.mean_diff:+.3f} ({q.n_positive}/{q.n_paired_seeds}, {ptab(q.paired_t_p, label=True)})")
        elif len(q):
            cells.append(f"{q.iloc[0].mean_diff:+.3f}")
        else:
            cells.append("--")
        rows.append([lab] + cells)
    hdr = ["Metric (internal test unless marked)", f"E0 ResNet-50 ($n$={nseeds.get('resnet50_E0', 0)})",
           f"E1 ResNet-50 ($n$={nseeds.get('resnet50_E1', 0)})", f"E1 EffNet-B3 ($n$={nseeds.get('effnetb3_E1', 0)})",
           "E1$-$E0 paired"]
    write("tab_multiseed", tabular(
        "Mean $\\pm$ SD over independently trained seeds (seed 42 = submitted run; further seeds retrained with the "
        "identical protocol). All post-hoc maps and thresholds fitted on validation. Last column: mean per-seed "
        "difference, with (number of seeds where E1 $>$ E0 / seeds paired, paired $t$-test $p$). "
        "$^\\dagger$Official ISIC 2019 test "
        "set, never used during development (8 known classes; UNK = out-of-distribution).",
        "tab:multiseed", hdr, rows, "lcccc", star=True))
    for mk, tag in [("resnet50_E0", "Zero"), ("resnet50_E1", "One"), ("effnetb3_E1", "Eff")]:
        m(f"NSeeds{tag}", nseeds.get(mk, 0))
        for met, nm in [("test_balanced_accuracy", "BalAcc"), ("test_macro_f1", "MacroFone"), ("test_accuracy", "Acc"),
                        ("test_ECE_afterT", "ECEafter"), ("test_AURC_MSP", "AURC"), ("test_AUROC_MSP", "AUROCmsp"),
                        ("test_AUROC_MC_entropy_deterr", "AUROCmc"), ("T", "T"), ("off_bal_acc_known", "OffBal"),
                        ("off_OOD_AUROC_MSP", "OffOOD"), ("test_NLL_afterT", "NLLafter")]:
            m(f"MS{tag}{nm}", ms(mk, met)[0])
    for met, nm in [("test_balanced_accuracy", "BalAcc"), ("test_macro_f1", "MacroFone"), ("test_accuracy", "Acc"),
                    ("test_AURC_MSP", "AURC"), ("test_ECE_afterT", "ECEafter"), ("test_NLL_afterT", "NLLafter"),
                    ("test_AUROC_MSP", "AUROCmsp")]:
        q = p[(p.comparison == "resnet50_E1 - resnet50_E0") & (p.metric == met)]
        if len(q):
            q = q.iloc[0]
            m(f"MSDiff{nm}", f"{q.mean_diff:+.3f}")
            m(f"MSDiff{nm}P", fp(q.paired_t_p))
            m(f"MSDiff{nm}Pos", f"{q.n_positive}/{q.n_paired_seeds}")
    pr_runs = pd.read_csv(MS / "per_run_metrics.csv")
    new = sorted(int(s) for s in pr_runs[pr_runs.protocol == "paper"].seed.unique() if s != 42)
    if not new:
        txt = "none yet"
    elif len(new) > 2 and new == list(range(new[0], new[-1] + 1)):
        txt = f"{new[0]}--{new[-1]}"
    else:
        txt = ", ".join(map(str, new[:-1])) + (" and " if len(new) > 1 else "") + str(new[-1])
    m("SeedsNew", txt)
    sel = MS / "validation_model_selection_per_seed.csv"
    if sel.exists():
        v = pd.read_csv(sel)
        m("NSelResnet", int((v.selected_on_val_macro_f1 == "resnet50_E1").sum()))
        m("NSelTotal", len(v))
    ens = MS / "deep_ensembles.csv"
    if ens.exists():
        e = pd.read_csv(ens)
        r = e[e.ensemble.str.startswith("resnet50_E1")]
        if len(r):
            r = r.iloc[0]
            m("EnsN", int(r.n_members))
            for c, nm in [("test_balanced_accuracy", "BalAcc"), ("test_ECE", "ECE"), ("test_NLL", "NLL"),
                          ("test_AUROC_ens_MSP", "AUROCmsp"), ("test_AUROC_ens_entropy", "AUROCent"),
                          ("test_AUROC_ens_MI", "AUROCmi"), ("test_AURC_ens_MSP", "AURC"),
                          ("off_OOD_AUROC_ens_MSP", "OODmsp"), ("off_OOD_AUROC_ens_MI", "OODmi"),
                          ("off_bal_acc_known", "OffBal"), ("test_accuracy", "Acc"), ("test_macro_f1", "MacroFone")]:
                m(f"Ens{nm}", r[c])
    runs_df = pd.read_csv(MS / "per_run_metrics.csv")
    e1 = runs_df[(runs_df.model == "resnet50_E1") & (runs_df.protocol == "paper")]
    if "test_VS_cwECE" in e1.columns and len(e1):
        m("MSOneCwECET", e1.test_ECE_classwise_afterT.mean())
        m("MSOneVScwECE", e1.test_VS_cwECE.mean())
        m("MSOneVSBal", e1.test_VS_balanced_accuracy.mean())
        m("MSOneVSwins", f"{int((e1.test_VS_cwECE < e1.test_ECE_classwise_afterT).sum())}/{len(e1)}")
    for mk, tag in [("resnet50_E0", "Zero"), ("resnet50_E1", "One"), ("effnetb3_E1", "Eff")]:
        tbr = runs_df[(runs_df.model == mk) & (runs_df.protocol == "true_best")]
        if not len(tbr):
            continue
        same = runs_df[(runs_df.model == mk) & (runs_df.protocol == "paper") & runs_df.seed.isin(tbr.seed)]
        for met, nm in [("test_balanced_accuracy", "BalAcc"), ("T", "T"), ("test_ECE_before", "ECEbefore"),
                        ("test_AURC_MSP", "AURC"), ("test_AUROC_MC_MI", "MI"), ("test_NLL_afterT", "NLLafter")]:
            m(f"TBpair{tag}True{nm}", tbr[met].mean())
            m(f"TBpair{tag}Stop{nm}", same[met].mean())
        m(f"TBpair{tag}N", len(tbr))
        merged = tbr.merge(same, on="seed", suffixes=("_tb", "_stop"))
        m(f"TBpair{tag}Ndiff", int((merged["T_tb"] - merged["T_stop"]).abs().gt(1e-6).sum()))
        m(f"TBpair{tag}Seeds", ", ".join(map(str, sorted(tbr.seed))))
    tb = s[s.protocol == "true_best"]
    if len(tb):
        for mk, tag in [("resnet50_E0", "Zero"), ("resnet50_E1", "One"), ("effnetb3_E1", "Eff")]:
            for met, nm in [("test_balanced_accuracy", "BalAcc"), ("T", "T"), ("test_ECE_before", "ECEbefore"),
                            ("test_AURC_MSP", "AURC"), ("test_macro_f1", "MacroFone")]:
                r = tb[(tb.model == mk) & (tb.metric == met)]
                if len(r):
                    r = r.iloc[0]
                    m(f"TB{tag}{nm}", f"{r['mean']:.3f}" + (f" $\\pm$ {r.sd:.3f}" if np.isfinite(r.sd) else ""))
                    m(f"TB{tag}N", int(r.n_seeds))


# ------------------------------------------------------------------ official test
def official():
    f = OFF / "classification_calibration_known_classes.csv"
    if not f.exists():
        return
    c = pd.read_csv(f)
    o = pd.read_csv(OFF / "ood_unk_detection.csv")
    s = pd.read_csv(OFF / "selective_under_shift.csv")
    ch = pd.read_csv(OFF / "challenge_style_balanced_accuracy.csv")
    cf = pd.read_csv(OFF / "conformal_under_shift.csv")
    mod = {"resnet50_E1": "ResNet-50 E1", "resnet50_E0": "ResNet-50 E0", "effnetb3_E1": "EffNet-B3 E1"}
    rows = []
    sub = "all_8238"
    for mk in ["resnet50_E0", "resnet50_E1", "effnetb3_E1"]:
        r = c[(c.model == mk) & (c.subset == sub)].iloc[0]
        oo = o[(o.model == mk) & (o.subset == sub)].set_index("score")
        ss = s[(s.model == mk) & (s.subset == sub) & (s.rule == "MSP_T>=tau_val85")].iloc[0]
        cc = ch[(ch.model == mk) & (ch.subset == sub)].set_index("rule")
        q = cf[(cf.model == mk) & (cf.subset == sub) & (cf.alpha == 0.1)].iloc[0]
        rows.append([mod[mk], f3(r.accuracy), f3(r.balanced_accuracy), f3(r.ECE_before), f3(r.ECE_afterT),
                     f3(oo.loc["MSP_T", "AUROC_UNK_vs_known"]), f3(oo.loc["MC_entropy_T", "AUROC_UNK_vs_known"]),
                     f3(oo.loc["MC_mutual_info_T", "AUROC_UNK_vs_known"]), f3(ss.coverage_known), f3(ss.accepted_acc_known),
                     f3(ss.UNK_accepted_frac), f3(cc.loc["never_predict_UNK", "bal_acc_9class_reject_as_UNK"]),
                     f3(cc.loc["MSP_T>=tau_val85", "bal_acc_9class_reject_as_UNK"]), f3(q.coverage_known)])
    write("tab_official", tabular(
        "Untouched evaluation on the official ISIC 2019 test set (8,238 images; 2,047 UNK). Temperature, threshold "
        "($\\tau$ for 85\\% validation coverage) and conformal quantile come from the internal validation split. "
        "Acc./Bal.\\ acc./ECE on the 6,191 known-class images; OOD = UNK vs.\\ known AUROC; Cov./Sel.\\ acc.\\ on known classes; "
        "OOD MC-H: MC-Dropout predictive entropy; UNK acc.: fraction of UNK images accepted; BA$_9$: challenge-style "
        "9-class balanced accuracy when UNK is never predicted; BA$_9$ rej.: the same with rejected images labelled "
        "UNK; Conf.: LAC coverage on known classes at 90\\% nominal.",
        "tab:official", ["Model", "Acc.", "Bal.\\ acc.", "ECE", "ECE$_T$", "OOD MSP", "OOD MC-H", "OOD MI", "Cov.",
                         "Sel.\\ acc.", "UNK acc.", "BA$_9$", "BA$_9$ rej.", "Conf."], rows, "l" + "c" * 13,
        star=True, size="\\scriptsize"))
    r = c[(c.model == "resnet50_E1") & (c.subset == sub)].iloc[0]
    m("OffNknown", int(r.n_known))
    m("OffNunk", int(r.n_UNK))
    m("OffBal", r.balanced_accuracy)
    m("OffAcc", r.accuracy)
    m("OffECEbefore", r.ECE_before)
    m("OffECEafter", r.ECE_afterT)
    m("OffNLLafter", r.NLL_afterT)
    oo = o[(o.model == "resnet50_E1") & (o.subset == sub)].set_index("score")
    m("OffOODmsp", oo.loc["MSP_T", "AUROC_UNK_vs_known"])
    m("OffOODmc", oo.loc["MC_entropy_T", "AUROC_UNK_vs_known"])
    m("OffOODmi", oo.loc["MC_mutual_info_T", "AUROC_UNK_vs_known"])
    ss = s[(s.model == "resnet50_E1") & (s.subset == sub) & (s.rule == "MSP_T>=tau_val85")].iloc[0]
    m("OffSelCov", ss.coverage_known)
    m("OffSelAcc", ss.accepted_acc_known)
    m("OffAccAll", ss.overall_acc_known)
    m("OffUNKacc", ss.UNK_accepted_frac)
    m("OffUNKaccPct", 100 * ss.UNK_accepted_frac, 1)
    m("OffMELall", ss.MEL_sens_all)
    m("OffMELacc", ss.MEL_sens_accepted)
    cc = ch[(ch.model == "resnet50_E1") & (ch.subset == sub)].set_index("rule")
    m("OffBAnine", cc.loc["never_predict_UNK", "bal_acc_9class_reject_as_UNK"])
    m("OffBAnineRej", cc.loc["MSP_T>=tau_val85", "bal_acc_9class_reject_as_UNK"])
    m("OffUNKrecall", cc.loc["MSP_T>=tau_val85", "UNK_recall"])
    r2 = c[(c.model == "resnet50_E1") & (c.subset == "scored_subset")].iloc[0]
    m("OffBalScored", r2.balanced_accuracy)
    cc2 = ch[(ch.model == "resnet50_E1") & (ch.subset == "scored_subset")].set_index("rule")
    m("OffBAnineRejScored", cc2.loc["MSP_T>=tau_val85", "bal_acc_9class_reject_as_UNK"])
    q = cf[(cf.model == "resnet50_E1") & (cf.subset == sub) & (cf.alpha == 0.1)].iloc[0]
    m("OffConfCov", q.coverage_known)
    m("OffConfSize", q.avg_set_size_known, 2)
    bj = OFF / "bootstrap_ci_resnet50_E1.json"
    if bj.exists():
        b = json.loads(bj.read_text())
        m("OffBalCI", f"[{b['bal_acc_known'][0]:.3f}, {b['bal_acc_known'][1]:.3f}]")
        m("OffOODmspCI", f"[{b['AUROC_UNK_MSP'][0]:.3f}, {b['AUROC_UNK_MSP'][1]:.3f}]")
        m("OffOODmiCI", f"[{b['AUROC_UNK_MI'][0]:.3f}, {b['AUROC_UNK_MI'][1]:.3f}]")
    meta = json.loads((OFF / "meta.json").read_text())
    m("OffNexcl", meta["n_excluded_near_duplicates"])
    r3 = c[(c.model == "resnet50_E1") & (c.subset == "all_minus_near_duplicates")].iloc[0]
    m("OffBalNoDup", r3.balanced_accuracy)


# ------------------------------------------------------------------ integrity
def integrity():
    f = INT / "integrity_summary.json"
    if not f.exists():
        return
    s = json.loads(f.read_text())
    m("IntNlesion", f"{s['n_with_lesion_id']:,}".replace(",", "{,}"))
    m("IntNnolesion", f"{s['n_without_lesion_id']:,}".replace(",", "{,}"))
    m("IntNgroups", f"{s['n_groups']:,}".replace(",", "{,}"))
    m("IntMultiGroups", f"{s['groups_with_multiple_images']:,}".replace(",", "{,}"))
    m("IntExactFile", s["exact_dup_groups_md5_file"])
    m("IntExactPix", s["exact_dup_groups_md5_pixels"])
    m("IntExactCross", s["exact_dup_groups_cross_split_md5_pixels"])
    m("IntCand", s["near_dup_candidates_internal"])
    m("IntCandCross", s["near_dup_candidates_cross_split"])
    m("IntStrongCross", s["near_dup_strong_cross_split"])
    m("IntStrongTestImgs", s.get("near_dup_strong_cross_split_test_images", 0))
    m("IntOffStrong", s["official_vs_internal_strong"])
    m("IntOffImgs", s["official_images_with_strong_match"])
    m("IntOffExact", s["exact_dup_internal_vs_official_md5_pixels"])
    if s.get("strict_rule_within_split_same_lesion_frac") is not None:
        m("IntWithinSameLesion", 100 * s["strict_rule_within_split_same_lesion_frac"], 0)
        m("IntWithinPairs", s["strict_rule_within_split_pairs"])
    m("IntNearIdentical", s.get("near_identical_cross_split_pairs", 0))
    m("IntLooseTestImgs", s.get("loose_cross_split_test_images", 0))
    m("IntLoosePairs", s.get("loose_cross_split_pairs", 0))
    src = pd.read_csv(INT / "source_by_split.csv", index_col=0)
    rows = []
    for name in ["HAM10000", "BCN_20000", "MSK"]:
        if name in src.index:
            r = src.loc[name]
            rows.append([name.replace("_", "\\_"), str(int(r.get("train", 0))), str(int(r.get("val", 0))), str(int(r.get("test", 0)))])
    cnt = pd.read_csv(ROOT / "data" / "splits" / "train.csv").label.value_counts()
    cv = pd.read_csv(ROOT / "data" / "splits" / "validation.csv").label.value_counts()
    ct = pd.read_csv(ROOT / "data" / "splits" / "test.csv").label.value_counts()
    rows.append("MIDRULE")
    for cl in CLASSES:
        rows.append([cl, str(cnt[cl]), str(cv[cl]), str(ct[cl])])
    rows.append("MIDRULE")
    rows.append(["Total", str(cnt.sum()), str(cv.sum()), str(ct.sum())])
    write("tab_counts", tabular("Images per source collection and per class in the lesion-grouped split.",
                                "tab:counts", ["", "Train", "Val.", "Test"], rows, "lrrr"))
    for name, tag in [("HAM10000", "HAM"), ("BCN_20000", "BCN"), ("MSK", "MSK")]:
        if name in src.index:
            m(f"Src{tag}", int(src.loc[name].sum()))


# ------------------------------------------------------------------ grad-cam
def gradcam():
    f = GC / "randomisation_test.csv"
    if not all((GC / n).exists() for n in ["randomisation_test.csv", "deletion_insertion.csv", "summary.json",
                                           "centre_bias_and_uncertainty_link.csv"]):
        return
    rnd = pd.read_csv(f)
    di = pd.read_csv(GC / "deletion_insertion.csv").set_index("saliency")
    cb = pd.read_csv(GC / "centre_bias_and_uncertainty_link.csv").set_index("layer")
    rows = []
    for _, r in rnd[rnd.layer == "conv3_paper"].iterrows():
        rows.append([f"up to {r.randomised_up_to}", f3(r.mean_spearman), f3(r.mean_top10_IoU)])
    write("tab_gradcam_rand", tabular(
        "Cascading model-parameter randomisation (300 random test images): similarity between Grad-CAM maps of the "
        "trained and the progressively randomised network. Values near 0 indicate model-sensitive maps.",
        "tab:gradcamrand", ["Randomised", "Spearman $\\rho$", "Top-10\\% IoU"], rows, "lcc"))
    lab = {"gradcam_conv3": "Grad-CAM (paper layer)", "gradcam_layer4": "Grad-CAM (layer4 output)",
           "mean_gradcam": "Dataset-mean Grad-CAM", "centre_gaussian": "Centre Gaussian prior", "random": "Random map"}
    rows = [[lab[k], f3(di.loc[k, "deletion_AUC(lower=better)"]), f3(di.loc[k, "insertion_AUC(higher=better)"])]
            for k in lab if k in di.index]
    write("tab_gradcam_delins", tabular(
        "Deletion/insertion faithfulness (300 random test images; probability of the predicted class as salient "
        "pixels are blurred out / restored in 5\\% steps). Lower deletion and higher insertion AUC are better.",
        "tab:gradcamdelins", ["Saliency", "Deletion AUC", "Insertion AUC"], rows, "lcc"))
    for k, nm in [("gradcam_conv3", "GC"), ("centre_gaussian", "Centre"), ("random", "Rand"), ("mean_gradcam", "Mean"),
                  ("gradcam_layer4", "GCfour")]:
        m(f"Del{nm}", di.loc[k, "deletion_AUC(lower=better)"])
        m(f"Ins{nm}", di.loc[k, "insertion_AUC(higher=better)"])
    r = rnd[(rnd.layer == "conv3_paper")]
    m("RandHeadRho", r[r.randomised_up_to == "head"].iloc[0].mean_spearman, 2)
    m("RandFourRho", r[r.randomised_up_to == "layer4"].iloc[0].mean_spearman, 2)
    m("RandAllRho", r[r.randomised_up_to == "layer1"].iloc[0].mean_spearman, 2)
    c = cb.loc["conv3_paper"]
    m("GCcentral", 100 * c["mean_mass_in_central_25pct_area"], 1)
    m("GCoffset", c["mean_centroid_offset(frac_of_half_width)"], 2)
    m("GCerrAUROC", 1 - c["AUROC_error_spatial_entropy"])  # score = map concentration (low entropy)
    m("GCrhoMC", c["spearman_entropyMap_vs_MCentropy"], 2)
    s = json.loads((GC / "summary.json").read_text())["paired_gradcam_vs_centre"]
    m("GCvsCentreDelP", fp(s['deletion_wilcoxon_p']))
    m("GCvsCentreInsP", fp(s['insertion_wilcoxon_p']))


# ------------------------------------------------------------------ derived analyses from dumps
def wilson(k, n, z=1.96):
    if n == 0:
        return np.nan, np.nan
    ph = k / n
    den = 1 + z * z / n
    c = (ph + z * z / (2 * n)) / den
    h = z * np.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n)) / den
    return c - h, c + h


def derived():
    from scripts.revision.common import cls_metrics, ece_equal_width
    from src.evaluation.selective import _threshold_for_coverage
    d = load_dump(PAPER_RUNS["resnet50_E1"])
    d0 = load_dump(PAPER_RUNS["resnet50_E0"])
    T = fit_temperature(d["val"]["logits"], d["val"]["y"])
    p = softmax(d["test"]["logits"] / T, 1)
    y = d["test"]["y"]
    pred = p.argmax(1)
    pred0 = d0["test"]["logits"].argmax(1)
    tau = _threshold_for_coverage(softmax(d["val"]["logits"] / T, 1).max(1), 0.85)
    acc = p.max(1) >= tau
    m("SelTestCovPct", 100 * acc.mean(), 1)
    m("SelTestAccPct", 100 * (pred[acc] == y[acc]).mean(), 1)
    m("EOneTestBalAccS", float(np.mean([(pred[y == c] == c).mean() for c in range(len(CLASSES))])))
    # the submitted version applied the rounded threshold 0.43; the revision uses the unrounded validation value
    sub = p.max(1) >= 0.43
    m("SubNacc", int(sub.sum()))
    m("SubNrej", int((~sub).sum()))
    m("SubNcorrRej", int((pred[~sub] == y[~sub]).sum()))
    m("SubCov", sub.mean(), 3)
    m("SubAcc", (pred[sub] == y[sub]).mean(), 3)
    # per-class with Wilson CI on recall
    rows = []
    for c, cl in enumerate(CLASSES):
        n = int((y == c).sum())
        k1 = int(((pred == c) & (y == c)).sum())
        k0 = int(((pred0 == c) & (y == c)).sum())
        prec = k1 / max(1, int((pred == c).sum()))
        lo, hi = wilson(k1, n)
        lo0, hi0 = wilson(k0, n)
        f1 = 2 * prec * (k1 / n) / (prec + k1 / n) if prec + k1 / n > 0 else 0
        rows.append([cl, str(n), f"{k0 / n:.2f} [{lo0:.2f}, {hi0:.2f}]", f"{prec:.2f}", f"{k1 / n:.2f} [{lo:.2f}, {hi:.2f}]", f"{f1:.2f}"])
        m(f"Rec{cl}", f"{k1 / n:.2f}")
        m(f"Rec{cl}CI", f"[{lo:.2f}, {hi:.2f}]")
    write("tab_perclass", tabular(
        "Per-class results on the internal test split (seed 42) with 95\\% Wilson intervals for recall.",
        "tab:perclass", ["Class", "$n$", "E0 recall", "E1 prec.", "E1 recall", "E1 F1"], rows, "lrcccc"))
    # per source
    df = pd.read_csv(ROOT / "data" / "splits" / "test.csv")
    lid = df["lesion_id"].fillna("")
    src = np.where(lid.str.startswith("HAM_"), "HAM10000", np.where(lid.str.startswith("BCN_"), "BCN\\_20000", "MSK"))
    rows = []
    for sname in ["HAM10000", "BCN\\_20000", "MSK"]:
        k = src == sname
        if k.sum() == 0:
            continue
        present = [c for c in range(8) if (y[k] == c).sum() >= 5]
        bal = np.mean([(pred[k & (y == c)] == c).mean() for c in present])
        rows.append([sname, str(int(k.sum())), f"{(pred[k] == y[k]).mean():.3f}", f"{bal:.3f}",
                     f"{ece_equal_width(y[k], p[k]):.3f}", f"{acc[k].mean():.3f}",
                     f"{(pred[k & acc] == y[k & acc]).mean():.3f}"])
        tag = {"HAM10000": "HAM", "BCN\\_20000": "BCN", "MSK": "MSK"}[sname]
        m(f"Src{tag}TestAcc", (pred[k] == y[k]).mean())
        m(f"Src{tag}TestECE", ece_equal_width(y[k], p[k]))
        m(f"Src{tag}TestCov", acc[k].mean())
    write("tab_source", tabular(
        "Internal-test performance of ResNet-50 E1 by source collection (temperature and $\\tau$ from validation). "
        "Balanced accuracy is averaged over classes with $\\ge$5 test images in that source (MSK: MEL, NV, BKL only). "
        "Cov./Sel.\\ acc.: coverage and accepted accuracy at the 85\\% validation target.",
        "tab:source", ["Source", "$n$", "Acc.", "Bal.\\ acc.", "ECE$_T$", "Cov.", "Sel.\\ acc."], rows,
        "lrccccc", star=False))
    # near-duplicate exclusion effect
    cand = INT / "near_duplicate_candidates_internal.csv"
    if cand.exists():
        cd = pd.read_csv(cand)
        cd = cd[cd.cross_split & cd.strong] if len(cd) else cd
        bad = set(cd[cd.split_a == "test"].a) | set(cd[cd.split_b == "test"].b) if len(cd) else set()
        keep = ~df.image_id.isin(bad).values
        m("NoDupN", int((~keep).sum()))
        cm = cls_metrics(y[keep], p[keep])
        m("NoDupBalAcc", cm["balanced_accuracy"])
        m("NoDupAcc", cm["accuracy"])
        m("NoDupECE", ece_equal_width(y[keep], p[keep]))
        m("NoDupSelAcc", (pred[keep & acc] == y[keep & acc]).mean())
        cl = pd.read_csv(cand)
        if "loose" in cl.columns:
            cl = cl[cl.cross_split & cl.loose]
            badl = set(cl[cl.split_a == "test"].a) | set(cl[cl.split_b == "test"].b)
            keepl = ~df.image_id.isin(badl).values
            m("NoDupLooseN", int((~keepl).sum()))
            m("NoDupLooseBalAcc", cls_metrics(y[keepl], p[keepl])["balanced_accuracy"])
            m("NoDupLooseSelAcc", (pred[keepl & acc] == y[keepl & acc]).mean())


def tta():
    f = OUT / "tta" / "tta_results.csv"
    if not f.exists():
        return
    t = pd.read_csv(f)
    r = t[(t.model == "resnet50_E1") & (t.split == "test")]
    o = t[(t.model == "resnet50_E1") & (t.split == "official_test")]
    if len(r):
        r = r.iloc[0]
        m("TTAerrEnt", r.AUROC_err_TTA_entropy)
        m("TTAerrMI", r.AUROC_err_TTA_MI)
        m("TTAbal", r.TTA_bal_acc)
    if len(o):
        o = o.iloc[0]
        m("TTAoodEnt", o.AUROC_UNK_TTA_entropy)
        m("TTAoodMI", o.AUROC_UNK_TTA_MI)


def _pm(v, nd=3):
    v = np.asarray(v, dtype=float)
    v = v[np.isfinite(v)]
    if not len(v):
        return "--"
    return f"{v.mean():.{nd}f}" + (f"$\\pm${v.std(ddof=1):.{nd}f}" if len(v) > 1 else "")


def compact():
    """Tables for the 6-page main text."""
    from scipy import stats as st
    f = MS / "per_run_metrics.csv"
    if not f.exists():
        return
    d = pd.read_csv(f)
    d = d[d.protocol == "paper"]
    cols = [("resnet50_E0", "E0", None), ("resnet50_E0", "E0+LA", "LA"), ("resnet50_E1", "E1", None),
            ("effnetb3_E1", "B3-E1", None), ("resnet50_E0_r320", "E0-320", None), ("resnet50_E1_r320", "E1-320", None)]
    cols = [c for c in cols if (d.model == c[0]).any()]
    rows_def = [("test_balanced_accuracy", "Balanced accuracy"), ("test_accuracy", "Accuracy"),
                ("test_macro_f1", "Macro-F1"), ("test_roc_auc_macro_ovr", "ROC-AUC"),
                ("test_ECE_before", "ECE, uncalibrated"), ("test_ECE_afterT", "ECE, after $T$"),
                ("test_NLL_afterT", "NLL, after $T$"), ("test_AUROC_MSP", "Error AUROC (MSP)"),
                ("test_AURC_MSP", "AURC (MSP)"), ("off_bal_acc_known", "Official: bal.\\ acc.$^\\dagger$"),
                ("off_OOD_AUROC_MSP", "Official: UNK AUROC$^\\dagger$")]
    la_map = {"test_balanced_accuracy": "test_LA_balanced_accuracy", "test_accuracy": "test_LA_accuracy",
              "test_macro_f1": "test_LA_macro_f1", "test_ECE_afterT": "test_LA_ECE_afterT",
              "test_NLL_afterT": "test_LA_NLL_afterT", "test_AUROC_MSP": "test_LA_AUROC_MSP",
              "test_AURC_MSP": "test_LA_AURC_MSP", "off_bal_acc_known": "off_LA_bal_acc_known"}
    rows = []
    for met, lab in rows_def:
        r = [lab]
        for mk, nm, kind in cols:
            g = d[d.model == mk]
            col = la_map.get(met) if kind == "LA" else met
            r.append(_pm(g[col]) if col and col in g.columns else "--")
        a = d[d.model == "resnet50_E1"].set_index("seed")[met]
        b = d[d.model == "resnet50_E0"].set_index("seed")[met]
        sd = sorted(set(a.index) & set(b.index))
        diff = (a.loc[sd] - b.loc[sd]).values
        p = st.ttest_1samp(diff, 0).pvalue if len(diff) > 1 else np.nan
        r.append(f"{diff.mean():+.3f} ({ptab(p)})" if len(diff) > 1 else f"{diff.mean():+.3f}")
        rows.append(r)
    ns = {nm: int((d.model == mk).sum()) for mk, nm, _ in cols}
    hdr = ["Metric"] + [f"{nm} ($n$={ns[nm]})" for _, nm, _ in cols] + ["E1$-$E0 ($p$)"]
    write("tab_main", tabular(
        "Internal-test results (mean$\\pm$SD over seeds) for ResNet-50 with plain (E0) or class-weighted (E1) "
        "cross-entropy, E0 with post-hoc logit adjustment (E0+LA), EfficientNet-B3 (B3-E1)"
        + (" and ResNet-50 at $320\\times320$ px" if any(c[0].endswith("r320") for c in cols) else "")
        + ". Temperatures, thresholds and logit adjustment use validation data only. Last column: mean paired "
        "per-seed difference between E1 and E0 at $224\\times224$ px (paired $t$-test $p$). $^\\dagger$Official ISIC 2019 test set (6,191 known-class and "
        "2,047 UNK images), never used during development.",
        "tab:main", hdr, rows, "l" + "c" * (len(cols) + 1), star=True))
    # paired LA vs E1 / LA vs E0
    a = d[d.model == "resnet50_E0"].set_index("seed")
    b = d[d.model == "resnet50_E1"].set_index("seed")
    sd = sorted(set(a.index) & set(b.index))
    for met, nm in [("balanced_accuracy", "BalAcc"), ("AURC_MSP", "AURC"), ("NLL_afterT", "NLL"),
                    ("ECE_afterT", "ECE"), ("AUROC_MSP", "AUROC")]:
        la = a.loc[sd, f"test_LA_{met}"].values
        e1 = b.loc[sd, f"test_{met}"].values
        e0 = a.loc[sd, f"test_{met}"].values
        m(f"LA{nm}", la.mean())
        m(f"LAvsOne{nm}", f"{(la - e1).mean():+.3f}")
        m(f"LAvsOne{nm}P", fp(st.ttest_1samp(la - e1, 0).pvalue) if len(sd) > 1 else "--")
        m(f"LAvsZero{nm}", f"{(la - e0).mean():+.3f}")
    m("LAnseeds", len(sd))
    m("LAoffBal", a.loc[sd, "off_LA_bal_acc_known"].mean())
    # compact calibration table (seed 42)
    cv = pd.read_csv(S42 / "D_calibration_variants_test.csv")
    e0s42 = d[(d.model == "resnet50_E0") & (d.seed == 42)].iloc[0]
    rows = []
    for mk, var, lab in [("resnet50_E1", "uncalibrated", "E1, none"), ("resnet50_E1", "temperature", "E1, temperature"),
                         ("resnet50_E1", "vector_scaling", "E1, vector scaling"),
                         ("resnet50_E1", "prior_correction+T", "E1, prior corr.\\,+\\,$T$"),
                         ("resnet50_E0", "temperature", "E0, temperature")]:
        r = cv[(cv.model == mk) & (cv.variant == var)].iloc[0]
        rows.append([lab, f3(r.ECE), f3(r.ECE_classwise), f3(r.NLL), f3(r.balanced_accuracy)])
    rows.append(["E0+LA, temperature", f3(e0s42.test_LA_ECE_afterT),
                 f3(e0s42.test_LA_cwECE) if "test_LA_cwECE" in e0s42 else "--", f3(e0s42.test_LA_NLL_afterT),
                 f3(e0s42.test_LA_balanced_accuracy)])
    write("tab_calib_compact", tabular(
        "Calibration on the internal test split (seed 42; all maps fitted on validation). cwECE: class-wise ECE.",
        "tab:calib", ["Model, post-hoc map", "ECE", "cwECE", "NLL", "Bal.\\ acc."], rows, "lcccc"))
    # compact score table
    e = pd.read_csv(S42 / "E_uncertainty_scores_test.csv")
    e = e[e.model == "resnet50_E1"].set_index("score")
    dd = pd.read_csv(S42 / "E_auroc_diff_vs_MSP_resnet50_E1.csv").set_index("score")
    oo = pd.read_csv(OFF / "ood_unk_detection.csv")
    oo = oo[(oo.model == "resnet50_E1") & (oo.subset == "all_8238")].set_index("score")["AUROC_UNK_vs_known"]
    rows = []

    def star(lo, hi):
        return "$^{*}$" if (lo > 0 or hi < 0) else ""
    for sc, lab in [("MSP_T", "MSP"), ("softmax_entropy_T", "Softmax entropy"),
                    ("MC_entropy_T", "MC entropy"), ("MC_mutual_info_T", "MC MI")]:
        r = e.loc[sc]
        dl = "--" if sc == "MSP_T" else f"{dd.loc[sc, 'diff_vs_MSP_T']:+.3f}{star(dd.loc[sc, 'ci_low'], dd.loc[sc, 'ci_high'])}"
        rows.append([lab, f3(r.AUROC_det_errors), dl, f3(r.AURC_det), f3(oo[sc])])
    tt = OUT / "tta" / "tta_table_rows_resnet50_E1.csv"
    tr = OUT / "tta" / "tta_results.csv"
    if tt.exists() and tr.exists():
        t1 = pd.read_csv(tt).set_index("score")
        t2 = pd.read_csv(tr)
        t2 = t2[(t2.model == "resnet50_E1") & (t2.split == "official_test")].iloc[0]
        for sc, lab, ood in [("TTA_entropy", "TTA entropy", t2.AUROC_UNK_TTA_entropy),
                             ("TTA_mutual_info", "TTA MI", t2.AUROC_UNK_TTA_MI)]:
            r = t1.loc[sc]
            rows.append([lab, f3(r.AUROC), f"{r.diff_vs_MSP_T:+.3f}{star(r.ci_low, r.ci_high)}", f3(r.AURC), f3(ood)])
    ens = MS / "deep_ensembles.csv"
    if ens.exists():
        en = pd.read_csv(ens)
        en = en[en.ensemble.str.startswith("resnet50_E1_paper")]
        if len(en):
            en = en.iloc[0]
            k = int(en.n_members)
            rows.append([f"Ens.\\ MSP ({k})", f3(en.test_AUROC_ens_MSP), "--", f3(en.test_AURC_ens_MSP),
                         f3(en.off_OOD_AUROC_ens_MSP)])
            rows.append([f"Ens.\\ MI ({k})", f3(en.test_AUROC_ens_MI), "--", "--", f3(en.off_OOD_AUROC_ens_MI)])
    write("tab_scores", tabular(
        "Error detection on the internal test split (Err.: AUROC; AURC) and UNK detection on the official test set "
        "(UNK: AUROC) for ResNet-50 E1, seed 42. $\\Delta$: paired AUROC difference to MSP; $^{*}$95\\% "
        "lesion-cluster bootstrap CI excludes 0. MC: head MC-Dropout; MI: mutual information (epistemic part); "
        "all scores use temperature-scaled logits (each MC pass is scaled before averaging). Ens.\\ ($k$): average of "
        "the $k$ ResNet-50 E1 seed models at $224\\times224$ px.",
        "tab:scores", ["Score", "Err.", "$\\Delta$", "AURC", "UNK"], rows, "lcccc"))
    # threshold transfer table (seed 42 E1): internal val -> internal test -> official test
    from src.evaluation.selective import _threshold_for_coverage
    dm = load_dump(PAPER_RUNS["resnet50_E1"])
    T = fit_temperature(dm["val"]["logits"], dm["val"]["y"])
    pv = softmax(dm["val"]["logits"] / T, 1)
    pt = softmax(dm["test"]["logits"] / T, 1)
    po = softmax(dm["official_test"]["logits"] / T, 1)
    yt, yo = dm["test"]["y"], dm["official_test"]["y"]
    kn = yo < 8
    rows = []
    for tgt in [0.70, 0.85, 0.95]:
        tau = _threshold_for_coverage(pv.max(1), tgt)
        at, ao = pt.max(1) >= tau, po.max(1) >= tau
        rows.append([f"{tgt:.2f}", f"{tau:.3f}", f3((pv.max(1) >= tau).mean()), f3(at.mean()),
                     f3((pt.argmax(1)[at] == yt[at]).mean()), f3(ao[kn].mean()),
                     f3((po.argmax(1)[ao & kn] == yo[ao & kn]).mean()), f3(ao[yo == 8].mean())])
    rows.append(["1.00", "--", "1.000", "1.000", f3((pt.argmax(1) == yt).mean()), "1.000",
                 f3((po.argmax(1)[kn] == yo[kn]).mean()), "1.000"])
    write("tab_transfer", tabular(
        "Validation-locked confidence thresholds (ResNet-50 E1, seed 42) applied to the internal and official test "
        "sets. Cov.: coverage; Acc.: accuracy on accepted images; UNK acc.: fraction of UNK images accepted.",
        "tab:transfer", ["Target", "$\\tau$", "Val.\\ cov.", "Test cov.", "Test acc.", "Off.\\ cov.", "Off.\\ acc.",
                         "UNK acc."], rows, "cccccccc", star=True, size="\\footnotesize"))


def r320():
    """Supplementary section + macros for the higher-resolution (320 px) replication, once available."""
    from scipy import stats as st
    f = MS / "per_run_metrics.csv"
    out = G / "supp_r320.tex"
    d = pd.read_csv(f) if f.exists() else pd.DataFrame()
    d = d[d.protocol == "paper"] if len(d) else d
    have = len(d) and (d.model == "resnet50_E1_r320").any()
    if not have:
        out.write_text("\\section{Higher-Resolution Replication}\n\\label{sec:s-r320}\n"
                       "\\textcolor{red}{[ResNet-50 E0/E1 at $320\\times320$ px: training in progress; this section is "
                       "generated automatically when the runs finish.]}\n", encoding="utf-8")
        return
    mets = [("test_balanced_accuracy", "Balanced accuracy"), ("test_accuracy", "Accuracy"),
            ("test_ECE_before", "ECE, uncalibrated"), ("test_ECE_afterT", "ECE, after $T$"),
            ("test_NLL_afterT", "NLL, after $T$"), ("T", "Temperature"),
            ("test_AUROC_MSP", "Error AUROC (MSP)"), ("test_AUROC_MC_entropy_deterr", "Error AUROC (MC entropy)"),
            ("test_AURC_MSP", "AURC (MSP)"), ("off_bal_acc_known", "Official bal.\\ acc."),
            ("off_OOD_AUROC_MSP", "Official UNK AUROC (MSP)"), ("off_OOD_AUROC_MC_MI", "Official UNK AUROC (MC MI)")]
    models = [("resnet50_E0", "E0 224"), ("resnet50_E1", "E1 224"), ("resnet50_E0_r320", "E0 320"),
              ("resnet50_E1_r320", "E1 320")]
    models = [x for x in models if (d.model == x[0]).any()]
    rows = []
    for met, lab in mets:
        rows.append([lab] + [_pm(d[d.model == mk][met]) for mk, _ in models])
    ns = {nm: int((d.model == mk).sum()) for mk, nm in models}
    tab = tabular("ResNet-50 at $224\\times224$ vs.\\ $320\\times320$ px (mean$\\pm$SD over seeds; internal test unless "
                  "stated). Same protocol; only the input resolution changes.", "tab:r320",
                  ["Metric"] + [f"{nm} ($n$={ns[nm]})" for _, nm in models], rows, "l" + "c" * len(models))
    e1 = d[d.model == "resnet50_E1_r320"]
    e0 = d[d.model == "resnet50_E0_r320"]
    m("RtwoOneBalAcc", e1.test_balanced_accuracy.mean())
    m("RtwoOneAUROCmsp", e1.test_AUROC_MSP.mean())
    m("RtwoOneAUROCmc", e1.test_AUROC_MC_entropy_deterr.mean())
    m("RtwoOneOffBal", e1.off_bal_acc_known.mean())
    m("RtwoN", len(e1))
    seeds = sorted(set(e1.seed))
    m("RtwoSeeds", " and ".join(str(s) for s in seeds) if len(seeds) <= 2 else ", ".join(map(str, seeds)))
    txt = ("\\section{Higher-Resolution Replication}\n\\label{sec:s-r320}\n"
           "To test whether the conclusions depend on the $224\\times224$ input, ResNet-50 E0 and E1 were retrained at "
           "$320\\times320$ (seeds \\RtwoSeeds) with otherwise identical settings (Table~\\ref{tab:r320}).\n\n" + tab)
    if len(e0):
        sd = sorted(set(e1.seed) & set(e0.seed))
        if sd:
            a = e1.set_index("seed").loc[sd]
            b = e0.set_index("seed").loc[sd]
            m("RtwoDiffBalAcc", f"{(a.test_balanced_accuracy - b.test_balanced_accuracy).mean():+.3f}")
            m("RtwoDiffAURC", f"{(a.test_AURC_MSP - b.test_AURC_MSP).mean():+.3f}")
            m("RtwoDiffNLL", f"{(a.test_NLL_afterT - b.test_NLL_afterT).mean():+.3f}")
            m("RtwoDiffAUROC", f"{(a.test_AUROC_MSP - b.test_AUROC_MSP).mean():+.3f}")
            m("RtwoDiffOOD", f"{(a.off_OOD_AUROC_MSP - b.off_OOD_AUROC_MSP).mean():+.3f}")
            m("RtwoNpair", len(sd))
            bal = [f"{v:+.3f}" for v in (a.test_balanced_accuracy - b.test_balanced_accuracy)]
            m("RtwoBalAccSeeds", ", ".join(bal[:-1]) + " and " + bal[-1] if len(bal) > 1 else bal[0])
            cost = ((a.test_NLL_afterT > b.test_NLL_afterT) & (a.test_AURC_MSP > b.test_AURC_MSP)
                    & (a.test_AUROC_MSP < b.test_AUROC_MSP) & (a.off_OOD_AUROC_MSP < b.off_OOD_AUROC_MSP))
            m("RtwoCostN", int(cost.sum()))
            m("RtwoZeroBalAcc", b.test_balanced_accuracy.mean())
            m("RtwoZeroOffBal", b.off_bal_acc_known.mean())
            # 320 px versus 224 px for the same seeds
            for mk, tag, sub in [("resnet50_E0", "Zero", b), ("resnet50_E1", "One", a)]:
                base = d[d.model == mk].set_index("seed").loc[sd]
                m(f"RtwoRes{tag}BalAcc", f"{(sub.test_balanced_accuracy - base.test_balanced_accuracy).mean():+.3f}")
                m(f"RtwoRes{tag}AURC", f"{(sub.test_AURC_MSP - base.test_AURC_MSP).mean():+.3f}")
                m(f"RtwoRes{tag}ECE", f"{(sub.test_ECE_afterT - base.test_ECE_afterT).mean():+.3f}")
            # post-hoc logit adjustment of the 320 px E0 model versus 320 px E1
            if "test_LA_balanced_accuracy" in b:
                m("RtwoLABalAcc", b.test_LA_balanced_accuracy.mean())
                for met, nm in [("ECE_afterT", "ECE"), ("AUROC_MSP", "AUROC"), ("AURC_MSP", "AURC")]:
                    m(f"RtwoLAvsOne{nm}", f"{(b[f'test_LA_{met}'] - a[f'test_{met}']).mean():+.3f}")
                txt += (
                    "\nWith \\RtwoNpair{} paired seeds this replication is small, so it is a consistency check rather "
                    "than a separate result. "
                    "Relative to the same seeds at $224\\times224$, the higher resolution changed balanced accuracy by "
                    "\\RtwoResZeroBalAcc{} (E0) and \\RtwoResOneBalAcc{} (E1) and AURC by \\RtwoResZeroAURC{} and "
                    "\\RtwoResOneAURC{}, while the calibrated ECE changed by \\RtwoResZeroECE{} and \\RtwoResOneECE. "
                    "The reliability cost of class weighting recurred in \\RtwoCostN{} of \\RtwoNpair{} seeds: after "
                    "temperature scaling, E1 had a higher NLL (\\RtwoDiffNLL), a larger AURC (\\RtwoDiffAURC), a lower "
                    "error AUROC (\\RtwoDiffAUROC) and a lower UNK detection AUROC on the official test set "
                    "(\\RtwoDiffOOD) than E0. Its balanced-accuracy gain was not consistent (per seed "
                    "\\RtwoBalAccSeeds). Logit adjustment of the $320\\times320$ E0 model reached a balanced accuracy "
                    "of \\RtwoLABalAcc{} (E1: \\RtwoOneBalAcc), with a lower ECE (\\RtwoLAvsOneECE) and a higher error "
                    "AUROC (\\RtwoLAvsOneAUROC) than E1 but a similar AURC (\\RtwoLAvsOneAURC).\n")
    out.write_text(txt, encoding="utf-8")


def ranking():
    f = MS / "summary_mean_sd.csv"
    if not f.exists():
        return
    s = pd.read_csv(f)
    s = s[s.protocol == "paper"]
    dims = [("test_balanced_accuracy", "Balanced accuracy", "max"), ("test_accuracy", "Accuracy", "max"),
            ("test_macro_f1", "Macro-F1", "max"), ("test_roc_auc_macro_ovr", "ROC-AUC", "max"),
            ("test_ECE_before", "ECE, uncalibrated", "min"), ("test_ECE_afterT", "ECE after $T$", "min"),
            ("test_NLL_afterT", "NLL after $T$", "min"), ("test_AUROC_MSP", "Error AUROC (MSP)", "max"),
            ("test_AURC_MSP", "AURC (MSP)", "min"), ("off_bal_acc_known", "Official test bal.\\ acc.", "max"),
            ("off_ECE_afterT", "Official test ECE$_T$", "min"), ("off_OOD_AUROC_MSP", "UNK detection AUROC", "max")]
    name = {"resnet50_E0": "R50-E0", "resnet50_E1": "R50-E1", "effnetb3_E1": "B3-E1"}
    rows = []
    for met, lab, how in dims:
        g = s[(s.metric == met) & s.model.isin(list(name))].set_index("model")["mean"]  # 224 px models only
        if not len(g):
            continue
        order = g.sort_values(ascending=(how == "min"))
        rows.append([lab] + [f"{g.get(k, np.nan):.3f}" for k in ["resnet50_E0", "resnet50_E1", "effnetb3_E1"]] +
                    [name[order.index[0]]])
    write("tab_ranking", tabular(
        "Which model is best depends on the reliability dimension (means over seeds; internal test unless stated).",
        "tab:ranking", ["Dimension", "R50-E0", "R50-E1", "B3-E1", "Best"], rows, "lcccc"))


def extras():
    """Numbers added after the independent audit of the response: per-seed validation metrics used for the
    architecture choice, MC entropy vs MSP across all runs, and the dermoscopic-vignette check."""
    f = MS / "per_run_metrics.csv"
    if f.exists():
        d = pd.read_csv(f)
        d = d[d.protocol == "paper"]
        a = d[d.model == "resnet50_E1"].set_index("seed")
        b = d[d.model == "effnetb3_E1"].set_index("seed")
        sd = sorted(set(a.index) & set(b.index))
        for met, nm in [("val_macro_f1", "Fone"), ("val_balanced_accuracy", "Bal")]:
            diffs = [f"{v:+.3f}" if round(v, 3) != 0 else "0.000" for v in (a.loc[sd, met] - b.loc[sd, met])]
            m(f"SelVal{nm}Diffs", ", ".join(diffs[:-1]) + " and " + diffs[-1])
        m("SelValSeeds", ", ".join(str(s) for s in sd[:-1]) + " and " + str(sd[-1]))
        diff = d.test_AUROC_MC_entropy_deterr - d.test_AUROC_MSP
        res = d.model.str.startswith("resnet50")
        m("MCbelowMSPres", int((diff[res] < 0).sum()))
        m("NrunsRes", int(res.sum()))
        eff = d.model == "effnetb3_E1"
        m("MCvsMSPeffMax", f"{diff[eff].abs().max():.3f}")
    vf = OUT / "artefacts" / "vignette_summary.csv"
    pf = OUT / "artefacts" / "vignette_prevalence_by_source.csv"
    if vf.exists() and pf.exists():
        v = pd.read_csv(vf)
        prev = pd.read_csv(pf).set_index("source")
        prev.columns = [c.strip() for c in prev.columns]
        m("VigBCN", 100 * prev.loc["BCN_20000", "True"], 0)
        m("VigHAM", 100 * prev.loc["HAM10000", "True"], 0)
        m("VigMSK", 100 * prev.loc["MSK", "True"], 0)
        per = pd.read_csv(OUT / "artefacts" / "vignette_per_image.csv")
        m("VigOff", 100 * per[per.split == "official_test"].vignette.mean(), 0)
        rows = []
        tag = {"internal test, all sources": "All", "internal test, BCN_20000 only": "BCN",
               "official test, known classes": "Off"}
        for scope in tag:
            for flag in [True, False]:
                r = v[(v.scope == scope) & (v.vignette == flag)].iloc[0]
                yn = "yes" if flag else "no"
                rows.append([scope.replace("BCN_20000", "BCN\\_20000") if flag else "", yn, str(int(r.n)),
                             f3(r.accuracy), f3(r.ECE_T), f3(r.coverage), f3(r.accepted_accuracy),
                             f3(r.error_AUROC_MSP), f3(r.UNK_share) if scope.startswith("official") else "--"])
                k = tag[scope] + ("Yes" if flag else "No")
                m(f"Vig{k}Acc", r.accuracy)
                m(f"Vig{k}AUROC", r.error_AUROC_MSP)
                m(f"Vig{k}Cov", r.coverage)
                if scope.startswith("official"):
                    m(f"Vig{k}UNK", 100 * r.UNK_share, 0)
        write("tab_vignette", tabular(
            "ResNet-50 E1 (seed 42) on images with and without a dark dermatoscope vignette (automatically detected: "
            "at least 3 of 4 corner patches with mean grey level below 35/255). Temperature and $\\tau$ (85\\% "
            "validation coverage) from validation. UNK share: fraction of UNK images among all official-test images "
            "in that group.", "tab:vignette",
            ["Images", "Vign.", "$n$", "Acc.", "ECE$_T$", "Cov.", "Sel.\\ acc.", "Err.\\ AUROC", "UNK share"], rows,
            "lcrcccccc"))


def main():
    seed42()
    multiseed()
    official()
    integrity()
    gradcam()
    derived()
    ranking()
    tta()
    compact()
    r320()
    extras()
    lines = ["% AUTO-GENERATED by scripts/revision/make_tex_assets.py -- do not edit by hand"]
    for k, v in sorted(macros.items()):
        assert k.isalpha(), k
        lines.append(f"\\newcommand{{\\{k}}}{{{v}}}")
    (G / "macros.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
    # fallbacks: any result macro used in main.tex that a stage has not produced yet renders as a
    # visible red marker instead of breaking compilation; missing generated tables get a stub.
    import re
    import shutil
    main_tex = (ROOT / "paper_revision" / "main.tex").read_text(encoding="utf-8")
    for extra in ["response_to_reviewers.tex", "supplementary.tex"]:
        f_extra = ROOT / "paper_revision" / extra
        if f_extra.exists():
            main_tex += f_extra.read_text(encoding="utf-8")
    latex_builtin = {"Delta", "LaTeX", "Large", "LARGE", "Huge", "Pr", "Rightarrow", "Leftrightarrow"}
    used = set(re.findall(r"\\([A-Z][A-Za-z]+)", main_tex)) - latex_builtin
    missing = sorted(u for u in used if u not in macros)
    fb = ["% AUTO-GENERATED: placeholders for results that are not available yet"]
    fb += [f"\\providecommand{{\\{k}}}{{\\textcolor{{red}}{{[{k}]}}}}" for k in missing]
    (G / "fallbacks.tex").write_text("\n".join(fb) + "\n", encoding="utf-8")
    for t in set(re.findall(r"\\input\{generated/(tab_[A-Za-z_]+)\}", main_tex)):
        if not (G / f"{t}.tex").exists():
            (G / f"{t}.tex").write_text(f"% pending: {t}\n\\textcolor{{red}}{{[table {t.replace('_', '-')} pending]}}\n",
                                        encoding="utf-8")
    # non-floating copies of every table for the one-column supplementary ([H] keeps them in reading order)
    for t in G.glob("tab_*.tex"):
        txt = t.read_text(encoding="utf-8").replace("\\begin{table*}[!t]", "\\begin{table}[H]") \
            .replace("\\end{table*}", "\\end{table}").replace("\\begin{table}[!t]", "\\begin{table}[H]")
        (G / f"supp_{t.name}").write_text(txt, encoding="utf-8")
    # copy figures used by the manuscript
    img = ROOT / "paper_revision" / "Images"
    for src in list((OUT / "figures").glob("*.png")) + list((OUT / "gradcam").glob("*.png")):
        shutil.copy2(src, img / src.name)
    print(f"{len(macros)} macros; {len(missing)} pending: {missing}")
    print(f"tables: {sorted(p.name for p in G.glob('tab_*.tex'))}")


if __name__ == "__main__":
    main()
