import pandas as pd
import wandb
from pathlib import Path
from src.utils import io_util
from src.utils import utils_helper


def export_latex_format(data_name, model_alias, model_name, n_layers, method, if_polysemy=True):
    api = wandb.Api()
    if model_name == "opt-1.3b":
        model_name = "opt-1"
    elif model_name == "opt-6.7b":
        model_name = "opt-6"

    METHOD = "Procrustes Analysis" if method == "procrustes" else "Ridge Regression"
    polysemy_type = "_polysemy" if if_polysemy else ""
    runs = api.runs(path=f"jalee/{data_name}-brain2{model_alias}_{method}_averaged{polysemy_type}", filters={"tags": model_name})

    MODEL_NAME = {'bert_uncased_L-2_H-128_A-2': "BERT$_{\\textsc{TINY}}$",
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

    MODEL_TAGS = {
        "ft": "fasttext",
        "bert": "bert",
        "gpt2": "gpt2",
        "opt": "opt"
    }

    data = []
    for i, run in enumerate(runs):
        summary = run.summary._json_dict
        try:
            metrics = {
                "P@1-CSLS": summary["precision_at_1-csls_knn_100"],
                "P@5-CSLS": summary["precision_at_5-csls_knn_100"],
                "P@10-CSLS": summary["precision_at_10-csls_knn_100"],
                "P@30-CSLS": summary["precision_at_30-csls_knn_100"],
                "P@50-CSLS": summary["precision_at_50-csls_knn_100"],
                "P@100-CSLS": summary["precision_at_100-csls_knn_100"],
                "P@1-NN": summary["precision_at_1-nn"],
                "P@5-NN": summary["precision_at_5-nn"],
                "P@10-NN": summary["precision_at_10-nn"],
                "P@30-NN": summary["precision_at_30-nn"],
                "P@50-NN": summary["precision_at_50-nn"],
                "P@100-NN": summary["precision_at_100-nn"]
            }
        except:
            print(i)
            print(summary)

        if model_alias in MODEL_TAGS:
            common_res = {
                "name": run.name,
                **metrics,
                "Layers": [i for i in run.tags if i.startswith("layer")][0],
                "Subjects": [i for i in run.tags if i.startswith("subject")][0],
                "Models": [i for i in run.tags if i.startswith(MODEL_TAGS[model_alias])][0]
            }
            if if_polysemy:
                common_res["Polysemy_bins"] = \
                [i for i in run.tags if i.startswith("2or3") or i.startswith("one") or i.startswith("over")][0]
            res = common_res
        else:
            print("Please choose the correct model result.")

        data.append(res)

    df = pd.DataFrame(data)
    # df['Subjects'] = df['Subjects'].replace({f'brain_{i}': f'Subject-{i}' for i in range(1, num_subs + 1)})
    df['Layers'] = df['Layers'].replace({f'layer_{i}': f'layer-{i}' for i in range(n_layers)})
    df['Models'] = df['Models'].replace(MODEL_NAME)

    group_names = ["Models", "Layers"]
    if if_polysemy:
        group_names.append("Polysemy_bins")

    precision_csls, precision_nn = [], []
    for k in [1, 5, 10, 30, 50, 100]:
        precision_csls.append(f'P@{k}-CSLS')
        precision_nn.append(f'P@{k}-NN')

    precision = precision_csls + precision_nn

    layer_avg = df.groupby(group_names)[precision].mean().reset_index()
    layer_avg = layer_avg.groupby(["Models"])[precision].max().reset_index()
    layer_avg.insert(loc=layer_avg.columns.get_loc("Models"), column="Summary",
                     value=[f"Ave Max."])

    layer_ = df.groupby(group_names.append("Subjects"))[precision].mean().reset_index()
    layer_max = layer_.groupby(["Models"])[precision].max().reset_index()
    layer_max.insert(loc=layer_max.columns.get_loc("Models"), column="Summary",
                     value=[f"Single Max."])

    layer_min = layer_.groupby(["Models"])[precision].min().reset_index()
    layer_min.insert(loc=layer_min.columns.get_loc("Models"), column="Summary",
                     value=[f"Single Min."])

    # Define your xltabular environment as a string
    xltabular_env = r"\begin{xltabular}{\linewidth}{ll|*{6}{>{\raggedleft\arraybackslash}X}|*{6}{>{\raggedleft\arraybackslash}X}}"
    xltabular_env += "\n" + r"\toprule" + "\n"
    xltabular_env += r"Subjects & Models & P@1-CSLS & P@5-CSLS & P@10-CSLS & P@30-CSLS & P@50-CSLS & P@100-CSLS & P@1-NN & P@5-NN & P@10-NN & P@30-NN & P@50-NN & P@100-NN \\ \midrule" + "\n"
    xltabular_env += r"\endfirsthead" + "\n"
    xltabular_env += r"\multicolumn{14}{c}%" + "\n"
    xltabular_env += r"{{\bfseries Table \thetable\ continued from previous page}} \\\midrule" + "\n"
    xltabular_env += r"Subjects & Models & P@1-CSLS & P@5-CSLS & P@10-CSLS & P@30-CSLS & P@50-CSLS & P@100-CSLS & P@1-NN & P@5-NN & P@10-NN & P@30-NN & P@50-NN & P@100-NN \\ \midrule" + "\n"
    xltabular_env += r"\endhead" + "\n"
    xltabular_env += r"\midrule" + "\n"
    xltabular_env += r"\multicolumn{14}{r}{{Continued on next page}} \\" + "\n"
    xltabular_env += r"\endfoot" + "\n"
    xltabular_env += r"\bottomrule" + "\n"
    xltabular_env += r"\endlastfoot" + "\n"

    # Generate your summary table
    total_summary = pd.concat([layer_avg, layer_max, layer_min], axis="index").reset_index(drop=True).round(2)

    # Style your table
    sum_styler = total_summary.style.hide(axis="index").format(precision=2).highlight_max(
        subset=precision, props='textbf:--rwrap;').highlight_min(subset=precision, props="color:{gray};")

    # Call the to_latex method on the Styler object and write to a file
    save_dir = Path(f"./latex_{data_name}_results")
    save_dir.mkdir(parents=True, exist_ok=True)

    with open(save_dir / f'{model_name}_{data_name}_{method}.txt', "w") as f:
        f.write(xltabular_env)
        f.write(
            sum_styler.to_latex().replace(
                "\\begin{tabular}", "").replace(
                "\\end{tabular}", "").replace(
                "\\begin{table}","").replace(
                "\\end{table}", "").replace(
                "{llrrrrrrrrrrrr}\nSummary & Models & P@1-CSLS & P@5-CSLS & P@10-CSLS & P@30-CSLS & P@50-CSLS & P@100-CSLS & P@1-NN & P@5-NN & P@10-NN & P@30-NN & P@50-NN & P@100-NN \\\\",
                "").strip())
        f.write("\n"r"\midrule"+"\n")
        f.write(r"\caption{"+f"{data_name} dataset: The results of {MODEL_NAME[model_name]} with {METHOD}." + "}\n")
        f.write(r"\label{"+f"tab:{method}_{data_name}-{model_name}"+"}\n")
        f.write(r"\end{xltabular}" + "\n")


    # Group by brain and layer and average P@k values
    # brain_layer_avg = df.groupby(['Subjects', "Models", 'Layers'])[precision].mean().reset_index().round(2)
    # total_summary = pd.concat([layer_avg, layer_max, layer_min], axis="index").reset_index(drop=True).round(2)
    #
    # sum_styler = total_summary.style.hide(axis="index").format(precision=2).highlight_max(
    #     subset=precision, props='textbf:--rwrap;').highlight_min(subset=precision, props="color:{gray};")
    # sum_styler.to_latex(f'./latex_results/{model_name}.txt', hrules=True,
    #                     caption=f"Harry Potter dataset: The results of {MODEL_NAME[model_name]} with Ridge Regression",
    #                     label=f"tab:regression_hp-{model_name}")


    # avg_styler = layer_avg.style.hide(axis="index").format(precision=2).highlight_max(
    #     subset=precision, props='textit:--rwrap; textbf:--rwrap;')
    # # Call the to_latex method on the Styler object
    # avg_styler.to_latex(f'./latex_results/{model_name}_avg.txt', hrules=True)
    #
    # brain_avg_styler = brain_layer_avg.style.hide(axis="index").format(precision=2).highlight_max(
    #     subset=precision, props='color:{green}; textbf:--rwrap;')
    # # brain_avg_styler.to_latex(f'./latex_results/{model_name}_single.txt', hrules=True)
    #
    # # add midrule after every n rows
    # for i in range(n_layers, len(brain_layer_avg), n_layers):
    #     brain_avg_styler = brain_avg_styler.set_properties(
    #         subset=pd.IndexSlice[i,"Subjects"],
    #         **{'midrule':""}
    #     )
    #
    # brain_avg_styler.to_latex(f'./latex_results/{model_name}_single.txt', hrules=True)



if __name__ == "__main__":
    parser = io_util.create_args_parser()
    args, unknown = parser.parse_known_args()
    config = io_util.load_config(args, unknown)
    utils_helper.enforce_reproducibility(seed=config.muse_parameters.seed)
    exp_dir = config.expdir.expname

    config = utils_helper.uniform_config(args=config)

    export_latex_format(config.data.dataset_name,
                        config.model.model_alias,
                        config.model.model_name,
                        config.model.n_layers,
                        method="procrustes",
                        if_polysemy=True)

