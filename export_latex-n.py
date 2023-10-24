import pandas as pd
import wandb
from pathlib import Path
from src.utils import io_util
from src.utils import utils_helper
import json

def export_latex_format(data_name, model_alias, method, extend_exp=""):
    api = wandb.Api()

    method_name = "Procrustes Analysis" if method == "procrustes" else "Ridge Regression"
    runs = api.runs(path=f"jalee/{data_name}-brain2{model_alias}_{method}_averaged-newtype{extend_exp}")

    model_name = {'bert_uncased_L-2_H-128_A-2': "BERT$_{\\textsc{TINY}}$",
                 'bert_uncased_L-4_H-256_A-4': "BERT$_{\\textsc{MINI}}$" ,
                 'bert_uncased_L-4_H-512_A-8': "BERT$_{\\textsc{SMALL}}$",
                 'bert_uncased_L-8_H-512_A-8': "BERT$_{\\textsc{MEDIUM}}$",
                 'bert-base-uncased': "BERT$_{\\textsc{BASE}}$",
                 'bert-large-uncased': "BERT$_{\\textsc{LARGE}}$",
                 'gpt2': "GPT2$_{\\textsc{BASE}}$",
                  'gpt2-medium': "GPT2$_{\\textsc{MEDIUM}}$",
                 'gpt2-large': "GPT2$_{\\textsc{LARGE}}$",
                 'gpt2-xl': "GPT2$_{\\textsc{XL}}$",
                 'opt-125m': "OPT$_{\\textsc{125M}}$",
                  'opt-1': "OPT$_{\\textsc{1.3B}}$",
                  'opt-6': "OPT$_{\\textsc{6.7B}}$",
                'opt-30b': "OPT$_{\\textsc{30B}}$",
                  "fasttext":"fastText"}

    model_tags = {
        "ft": "fasttext",
        "bert": "bert",
        "gpt2": "gpt2",
        "opt": "opt"
    }

    # metrics = runs.summary["Results"]

    # data = []
    df = pd.DataFrame()
    for i, single_run in enumerate(runs):
        metric = json.load(single_run.file(single_run.summary["Results"]["path"]).download(exist_ok=True))
        df = pd.concat([df,pd.DataFrame(metric["data"], columns=metric["columns"])])

    # # df['Subjects'] = df['Subjects'].replace({f'brain_{i}': f'Subject-{i}' for i in range(1, num_subs + 1)})
    # df['Layers'] = df['Layers'].replace({f'layer_{i}': f'layer-{i})
    df['Models'] = df['Models'].replace(model_name)
    #
    group_names = ["Models", "Layers"]
    if extend_exp != "":
        group_names.append("Bins")
    #
    precision_csls, precision_nn = [], []
    for k in [1, 5, 10, 30, 50, 100]:
        precision_csls.append(f'P@{k}-CSLS')
        precision_nn.append(f'P@{k}-NN')
    #
    precision = precision_csls + precision_nn
    #
    layer_avg = df.groupby(group_names)[precision].mean().reset_index()
    layer_avg = layer_avg.groupby(["Models"])[precision].max().reset_index()
    layer_avg.insert(loc=layer_avg.columns.get_loc("Models"), column="Summary",
                     value=[f"Ave Max."])

    #
    layer_ = df.groupby(group_names.append("Subjects"))[precision].mean().reset_index()
    layer_max = layer_.groupby(["Models", "Layers"])[precision].max().reset_index()
    # layer_max.insert(loc=layer_max.columns.get_loc("Models"), column="Summary",
    #                  value=[f"Single Max."])
    #
    # layer_min = layer_.groupby(["Models"])[precision].min().reset_index()
    # layer_min.insert(loc=layer_min.columns.get_loc("Models"), column="Summary",
    #                  value=[f"Single Min."])
    #
    # # Define your xltabular environment as a string
    # xltabular_env = r"\begin{xltabular}{\linewidth}{ll|*{6}{>{\raggedleft\arraybackslash}X}|*{6}{>{\raggedleft\arraybackslash}X}}"
    # xltabular_env += "\n" + r"\toprule" + "\n"
    # xltabular_env += r"Subjects & Models & P@1-CSLS & P@5-CSLS & P@10-CSLS & P@30-CSLS & P@50-CSLS & P@100-CSLS & P@1-NN & P@5-NN & P@10-NN & P@30-NN & P@50-NN & P@100-NN \\ \midrule" + "\n"
    # xltabular_env += r"\endfirsthead" + "\n"
    # xltabular_env += r"\multicolumn{14}{c}%" + "\n"
    # xltabular_env += r"{{\bfseries Table \thetable\ continued from previous page}} \\\midrule" + "\n"
    # xltabular_env += r"Subjects & Models & P@1-CSLS & P@5-CSLS & P@10-CSLS & P@30-CSLS & P@50-CSLS & P@100-CSLS & P@1-NN & P@5-NN & P@10-NN & P@30-NN & P@50-NN & P@100-NN \\ \midrule" + "\n"
    # xltabular_env += r"\endhead" + "\n"
    # xltabular_env += r"\midrule" + "\n"
    # xltabular_env += r"\multicolumn{14}{r}{{Continued on next page}} \\" + "\n"
    # xltabular_env += r"\endfoot" + "\n"
    # xltabular_env += r"\bottomrule" + "\n"
    # xltabular_env += r"\endlastfoot" + "\n"
    #
    # # Generate your summary table
    # total_summary = pd.concat([layer_avg, layer_max, layer_min], axis="index").reset_index(drop=True).round(2)
    #
    # # Style your table
    # sum_styler = total_summary.style.hide(axis="index").format(precision=2).highlight_max(
    #     subset=precision, props='textbf:--rwrap;').highlight_min(subset=precision, props="color:{gray};")
    #
    # # Call the to_latex method on the Styler object and write to a file
    # save_dir = Path(f"./latex_{data_name}_results")
    # save_dir.mkdir(parents=True, exist_ok=True)
    #
    # with open(save_dir / f'{model_name}_{data_name}_{method}.txt', "w") as f:
    #     f.write(xltabular_env)
    #     f.write(
    #         sum_styler.to_latex().replace(
    #             "\\begin{tabular}", "").replace(
    #             "\\end{tabular}", "").replace(
    #             "\\begin{table}","").replace(
    #             "\\end{table}", "").replace(
    #             "{llrrrrrrrrrrrr}\nSummary & Models & P@1-CSLS & P@5-CSLS & P@10-CSLS & P@30-CSLS & P@50-CSLS & P@100-CSLS & P@1-NN & P@5-NN & P@10-NN & P@30-NN & P@50-NN & P@100-NN \\\\",
    #             "").strip())
    #     f.write("\n"r"\midrule"+"\n")
    #     f.write(r"\caption{"+f"{data_name} dataset: The results of {MODEL_NAME[model_name]} with {METHOD}." + "}\n")
    #     f.write(r"\label{"+f"tab:{method}_{data_name}-{model_name}"+"}\n")
    #     f.write(r"\end{xltabular}" + "\n")


if __name__ == "__main__":
    export_latex_format(data_name="potter", model_alias="bert", method="procrustes", extend_exp="")
