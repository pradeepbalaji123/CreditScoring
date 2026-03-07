import shap
def compute_shap(model, background_data, sample):
    explainer = shap.DeepExplainer(model, background_data)
    shap_values = explainer.shap_values(
        sample,
        check_additivity=False
    )
    return shap_values