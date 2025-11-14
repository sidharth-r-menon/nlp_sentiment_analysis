import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st
import pandas as pd
import torch
import pickle
import pathlib
from src.evaluation.inference_preprocessor import InferencePreprocessor
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import streamlit.components.v1 as components


print("STARTING APP.PY")

css_path = pathlib.Path("assets/styles.css")
with open(css_path) as f:
        st.html(f"<style>{f.read()}</style>")

st.header("Sentiment Prediction")

st.sidebar.header("Model Settings")

def initialize_session_state():
    if "model_data" not in st.session_state:
        print("oh no model data run again")
        st.session_state.model_data = None
    if "sentiment_result" not in st.session_state:
        print("oh no sentiment_result run again")
        st.session_state.sentiment_result = None
    if "preprocessor" not in st.session_state:
        print("oh no preprocessor run again")
        st.session_state.preprocessor = InferencePreprocessor()
        st.session_state.preprocessor._initialize_nltk()
    if "model_info" not in st.session_state:
        st.session_state.model_info = None
    if "is_ensemble" not in st.session_state:
        st.session_state.is_ensemble = False

initialize_session_state()

model_type = st.sidebar.selectbox(
    "Select A Model Type",
    ["Transformer", "Stacking Ensemble", "Baseline"],
    help="Choose between baseline, transformer models, ensemble (combines all domains)",
)

if model_type == "Transformer":
    transformer_model = st.sidebar.selectbox(
        "Select A Transformer Model",
        ["BERT", "RoBERTa", "DistilBERT"],
    )
    model_selected = transformer_model.lower()
    
    train_domain = st.sidebar.selectbox(
        "Select A Training Domain",
        ["IMDb", "Amazon", "Yelp"],
        help="Domain the model was trained on",
    )
elif model_type == "Stacking Ensemble":
    transformer_model = st.sidebar.selectbox(
        "Select A Base Model Type",
        ["BERT", "RoBERTa", "DistilBERT"],
        help="Type of transformer used in ensemble"
    )
    model_selected = transformer_model.lower()
    train_domain = "ensemble" 
else:
    model_name = "baseline"
    train_domain = st.sidebar.selectbox(
        "Select A Training Domain",
        ["IMDb", "Amazon", "Yelp"],
        help="Domain the model was trained on"
    )

def load_model():
    st.session_state.sentiment_result = None
    st.session_state.model_info = None
    if model_type == "Transformer":
        model_path = f"models/saved_models/{transformer_model}_{train_domain}.pt".lower()
        st.session_state.model_data = torch.load(model_path, map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        st.session_state.preprocessor.set_transformer_tokenizer(model_name=st.session_state.model_data['model_name'], model_state_dict=st.session_state.model_data['model_state_dict'])
        st.session_state.is_ensemble = False
        model_infos = {
            "hint" : f"🤖 Current Model: {transformer_model} trained on {train_domain} reviews",
            'model_type' : "Transformer",
            'model' : transformer_model,
            'train_domain' : train_domain.lower()
        }
        st.session_state.model_info = model_infos

    elif model_type == "Stacking Ensemble":
        model_path = f"models/saved_models/{transformer_model}_ensemble.pkl".lower()
        with open(model_path, 'rb') as f:
            ensemble_data = pickle.load(f)
            st.session_state.preprocessor.set_ensemble_data(ensemble_data)
        domains = ['imdb', 'amazon', 'yelp']
        for domain in domains:
            model_path = f"models/saved_models/{transformer_model}_{domain}.pt".lower() #fix
            domain_model_data = torch.load(model_path, map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
            st.session_state.preprocessor.add_base_model(domain, model_name=domain_model_data['model_name'], model_state_dict=domain_model_data['model_state_dict'])
        st.session_state.preprocessor.set_tokenizer(model_name=domain_model_data['model_name'])
        st.session_state.is_ensemble = True
        model_infos = {
            "hint" : f"🤖 Current Model: {transformer_model} Stacking Ensemble (combines IMDb + Amazon + Yelp)",
            'model_type' : "Stacking Ensemble",
            'model' : transformer_model,
            'train_domain' : "all"
        }
        st.session_state.model_info = model_infos

    else:
        model_path = f"models/saved_models/{model_name}_{train_domain}.pkl".lower()
        with open(model_path, "rb") as f:
            st.session_state.model_data = pickle.load(f)
        st.session_state.is_ensemble = False
        model_infos = {
            "hint" : f"🤖 Current Model: Baseline (TF-IDF + Logistic Regression) trained on {train_domain} reviews",
            'model_type' : "Baseline",
            'model' : model_name,
            'train_domain' : train_domain.lower()
        }
        st.session_state.model_info = model_infos

st.sidebar.button("⏳ Load Model", type="primary", key="loadmodelbutton", on_click=load_model)
if "model_info" in st.session_state and st.session_state.model_info is not None:
    st.info(st.session_state.model_info['hint'])


try_tab, evalutaion_tab = st.tabs(["Try on Your Own", "Evaluation Results"])

with try_tab:
    user_input = st.text_area(
        "Enter your review:",
        placeholder="😎 LET'S TRY OUT SOME REVIEWS",
        key="querybox"
    )

    if "querybox" not in st.session_state:
        st.session_state.querybox = ""

    st.markdown(
        """
        <p style="
            font-family: Determination Mono, Determination Sans, monospace, ui-sans-serif, system-ui, sans-serif, Apple Color Emoji, Segoe UI Emoji, Segoe UI Symbol, Noto Color Emoji;
            font-size: 16px;
            font-weight: bold;
            text-align: left;
            padding: 0px 5px;
        ">
            No Idea ? 🤔 Try something below.
        </p>
        """,
        unsafe_allow_html=True
    )

    def set_example_value1():
        st.session_state.querybox =  "This movie was absolutely great! I love Astrid she is beautiful and brave."

    def set_example_value2():
        st.session_state.querybox = "I never had a product so bad in my life. The quality is worse. It doesn't match the pictures and description."

    def set_example_value3():
        st.session_state.querybox = "HeyTea tastes delicious! My favorite drink is bubble milk tea. Although, I think it's too sweet and expensive."

    tryout_col1, tryout_col2, tryout_col3 = st.columns(3)
    tryout_col1.button("Movie Reviews (Postive)", key="examplebutton1", on_click=set_example_value1)
    tryout_col2.button("Product Reviews (Negative)", key="examplebutton2", on_click=set_example_value2)
    tryout_col3.button("Restaurant Reviews (Mixed)", key="examplebutton3", on_click=set_example_value3)

    def anaylaze_sentiment():
        if user_input == "":
            st.warning("⚠️ Please enter some text to analyze.")
        else:
            texts = st.session_state.preprocessor.preprocess_text(user_input)
            if model_type == "Baseline":
                X = st.session_state.model_data['vectorizer'].transform([texts])
                prediction = st.session_state.model_data['classifier'].predict(X)[0]
                probabilities = st.session_state.model_data['classifier'].predict_proba(X)[0]
                confidence = probabilities[prediction]

                feature_names = st.session_state.model_data['vectorizer'].get_feature_names_out()
                coefficients = st.session_state.model_data['classifier'].coef_[0]
                feature_index = X.nonzero()[1]
                word_contributions = {feature_names[i]: coefficients[i] * X[0, i] for i in feature_index}
                word_contributions = dict(sorted(word_contributions.items(), key=lambda x: abs(x[1]), reverse=True))

                st.session_state.sentiment_result = {
                "prediction": prediction,
                "confidence": confidence,
                "attentsion": word_contributions,
                "base_probs": None
                }

            elif model_type == "Transformer":
                encodings = st.session_state.preprocessor.encode_texts([texts])
                with torch.no_grad():
                    input_ids = encodings['input_ids']
                    attention_mask = encodings['attention_mask']
                    predictions, probabilities, attentions = st.session_state.preprocessor.token_predict(input_ids, attention_mask)
                    prediction = predictions[0].item()
                    probability = probabilities[0]
                    confidence = probability[prediction].item() 

                st.session_state.sentiment_result = {
                    "prediction": prediction,
                    "confidence": confidence,
                    "attentsion": attentions,
                    "base_probs": None
                } 

            else:
                encodings = st.session_state.preprocessor.encode_texts([texts])
                with torch.no_grad():
                    input_ids = encodings['input_ids']
                    attention_mask = encodings['attention_mask']

                    dataset = TensorDataset(
                        input_ids,
                        attention_mask,
                        torch.tensor([0])
                    )
                    loader = DataLoader(dataset, batch_size=1)
                    results = st.session_state.preprocessor.ensemble_predict(loader)
                    ensemble_prediction = results['predictions'][0]
                    ensemble_probability = results['probabilities'][0]
                    ensemble_confidence = (ensemble_probability if ensemble_prediction == 1 else (1 - ensemble_probability))
                    
                    base_probs = {}
                    for domain in ['imdb', 'amazon', 'yelp']:
                        predictions, probabilities, _ = st.session_state.preprocessor.ensemble_base_predictions(input_ids, attention_mask, domain)
                        prediction = predictions[0].item()
                        probability = probabilities[0]
                        confidence = probability[prediction].item() 
                        base_probs[domain] = {
                        "prediction": prediction,
                        "confidence": confidence
                        } 

                    st.session_state.sentiment_result = {
                    "prediction": ensemble_prediction,
                    "confidence": ensemble_confidence,
                    "base_probs": base_probs,
                } 


    st.button("🔍 Analyze Sentiment", type="primary", key="analyzebutton", on_click=anaylaze_sentiment)
    st.markdown("---")

    if "sentiment_result" in st.session_state and st.session_state.sentiment_result is not None:
        st.markdown("#### Sentiment Prediction Result")
        result_col1, result_col2 = st.columns(2)
        with result_col1:
            sentiment = "Positive 😊" if st.session_state.sentiment_result["prediction"] == 1 else "Negative 😞"
            st.metric("Predicted Sentiment", sentiment)
        with result_col2:
            st.metric("Confidence", f"{st.session_state.sentiment_result['confidence']:.2%}")

        st.markdown("#### Word Influence on Prediction")
        if st.session_state.model_info['model_type'] == "Baseline":
            if not st.session_state.sentiment_result["attentsion"]:
                st.markdown("#### No significant words found in this input.")
            else:
                highlighted_text = ""
                for word in user_input.split():
                    clean_input = word.lower().strip(".,!?")
                    if clean_input in st.session_state.sentiment_result["attentsion"]:
                        val = st.session_state.sentiment_result["attentsion"][clean_input]
                        color = "green" if val > 0 else "red"
                        alpha = min(0.2 + abs(val) * 2, 0.9) 
                        highlighted_text += f"<span style='background-color:{color}; opacity:{alpha}; color:white; padding:2px 4px; border-radius:3px'>{word}</span> "
                    else:
                        highlighted_text += word + " "
                st.markdown(highlighted_text, unsafe_allow_html=True)

        if st.session_state.model_info['model_type'] == "Transformer":
            texts = st.session_state.preprocessor.preprocess_text(user_input)
            html = st.session_state.preprocessor.get_shap_html([texts])
            components.html(html, height=600, scrolling=True)

    if st.session_state.is_ensemble == True and "sentiment_result" in st.session_state and st.session_state.sentiment_result is not None and st.session_state.sentiment_result["base_probs"] is not None:
        st.markdown("---")
        st.markdown("#### Base Model Predictions:")
        base_probs = st.session_state.sentiment_result["base_probs"]
        ensemble_base_col1, ensemble_base_col2, ensemble_base_col3 = st.columns(3)
        with ensemble_base_col1:
            sentiment = "Positive 😊" if base_probs['imdb']["prediction"] == 1 else "Negative 😞"
            st.markdown("##### IMDb Model")
            st.metric("Predicted Sentiment", sentiment)
            st.metric("Confidence", f"{base_probs['imdb']['confidence']:.2%}")
        with ensemble_base_col2:
            sentiment = "Positive 😊" if base_probs['amazon']["prediction"] == 1 else "Negative 😞"
            st.markdown("##### Amazon Model")
            st.metric("Predicted Sentiment", sentiment)
            st.metric("Confidence", f"{base_probs['amazon']['confidence']:.2%}")
        with ensemble_base_col3:
            sentiment = "Positive 😊" if base_probs['yelp']["prediction"] == 1 else "Negative 😞"
            st.markdown("##### Yelp Model")
            st.metric("Predicted Sentiment", sentiment)  
            st.metric("Confidence", f"{base_probs['yelp']['confidence']:.2%}")

with evalutaion_tab:
    if st.session_state.model_info != None:

        if st.session_state.model_info['model_type'] == "Stacking Ensemble":
            df = pd.read_csv(rf"metrics/{st.session_state.model_info['model'].lower()}_ensemble_results.csv")

            st.markdown("#### Metric Comparison Across Test Domains")
            st.dataframe(df)

            st.markdown("#### Quick Insights")
            col1, col2, col3 = st.columns(3)
            col1.metric("Average Ensemble F1", f"{df['ensemble_f1'].mean():.3f}")
            best_domain = df.loc[df['ensemble_f1'].idxmax(), 'test_domain']
            col2.metric("Best Performing Domain", best_domain)
            avg_improvement = df['improvement'].mean()
            col3.metric("Average Improvement", f"{avg_improvement:+.3f}")
            st.divider()

            st.markdown("#### Ensemble Metrics Comparison")
            metrics = ["ensemble_accuracy", "ensemble_precision", "ensemble_recall", "ensemble_f1", "ensemble_roc_auc"]
            labels = df['test_domain']
            num_vars = len(metrics)
            angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
            angles += angles[:1] 
            fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))

            for i, domain in enumerate(labels):
                values = df.loc[i, metrics].tolist()
                values += values[:1] 
                ax.plot(angles, values, label=domain, linewidth=2)
                ax.fill(angles, values, alpha=0.1)
            ax.set_xticks(angles[:-1])
            ax.set_xticklabels([m.replace('ensemble_', '').replace('_', ' ').title() for m in metrics], fontsize=10)
            ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
            st.pyplot(fig)

            st.markdown("#### Ensemble Improvement Over Base Model")
            colors = ['green' if x > 0 else 'red' for x in df['improvement']]
            fig, ax = plt.subplots(figsize=(7,4))
            ax.bar(df['test_domain'], df['improvement'], color=colors)
            ax.axhline(0, color='black', linewidth=0.8)
            ax.set_ylabel('Improvement (ΔF1)')
            st.pyplot(fig)

            st.markdown("#### F1 scores Cross-Domain Generalization")
            cross_domain = df[['test_domain','imdb_f1','amazon_f1','yelp_f1']]
            cross_domain = cross_domain.set_index('test_domain')
            fig, ax = plt.subplots()
            sns.heatmap(cross_domain, annot=True, cmap="YlGnBu", ax=ax, fmt=".3f")
            st.pyplot(fig)

            st.markdown("#### Ensemble vs Base Model F1")
            x = np.arange(len(df['test_domain']))
            width = 0.35 
            fig, ax = plt.subplots(figsize=(7,4))
            ax.bar(x - width/2, df['ensemble_f1'], width, label='Ensemble F1', color='steelblue', alpha=0.8)
            ax.bar(x + width/2, df['best_base_f1'], width, label='Base Model F1', color='orange', alpha=0.8)
            ax.set_ylabel('F1 Score')
            ax.set_xlabel('Test Domain')
            ax.set_xticks(x)
            ax.set_xticklabels(df['test_domain'])
            ax.legend()
            ax.set_ylim(0, 1.05) 
            ax.set_title("Ensemble vs Base F1 by Domain")
            st.pyplot(fig)

        else:
            df = pd.read_csv(rf"metrics/{st.session_state.model_info['model'].lower()}_results.csv")
            current_train_domain = st.session_state.model_info['train_domain']
            selected_df = df[df["train_domain"] == current_train_domain]

            st.markdown(f"### 📊 {st.session_state.model_info['model'].capitalize()} Model on {current_train_domain} Evaluation Results Data")
            st.markdown(f"#### Dataframe")
            st.dataframe(selected_df)

            st.markdown("#### Metric Comparison Across Test Domains")

            metrics = ["accuracy", "precision", "recall", "f1", "roc_auc", "specificity", "sensitivity"]
            melted = df.melt(
                id_vars=["test_domain"],
                value_vars=metrics,
                var_name="Metric",
                value_name="Score"
            )

            fig, ax = plt.subplots(figsize=(9,5))
            sns.barplot(data=melted, x="Metric", y="Score", hue="test_domain", palette="viridis", ax=ax)
            ax.set_ylim(0, 1)
            plt.xticks(rotation=45)
            st.pyplot(fig)

            st.markdown("#### Radar Chart Comparison Across Test Domains")

            angles = np.linspace(0, 2*np.pi, len(metrics), endpoint=False).tolist()
            angles += angles[:1]

            fig = plt.figure(figsize=(6,6))
            ax = plt.subplot(111, polar=True)

            for _, row in selected_df.iterrows():
                values = [row[m] for m in metrics]
                values += values[:1]
                ax.plot(angles, values, label=row["test_domain"])
                ax.fill(angles, values, alpha=0.1)

            ax.set_xticks(angles[:-1])
            ax.set_xticklabels(metrics)
            ax.set_yticks(np.linspace(0, 1, 5))
            plt.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))
            st.pyplot(fig)

            st.markdown("#### Confusion Matrix of Different Test Domains")

            confusion_matrix_col1 = st.columns(2)
            confusion_matrix_col2 = st.columns(2)
            for i, (idx, row) in enumerate(selected_df.iterrows()):
                tn, fp, fn, tp = row["tn"], row["fp"], row["fn"], row["tp"]
                conf = np.array([[tn, fp],
                                [fn, tp]])
                fig, ax = plt.subplots()
                sns.heatmap(conf, annot=True, fmt='.0f', cmap='Blues', cbar=False, ax=ax)
                ax.set_xlabel("Predicted")
                ax.set_ylabel("Actual")
                ax.set_title(f"{row['test_domain']} Domain\nAccuracy={row['accuracy']:.3f}")
                if i < 2:
                    confusion_matrix_col1[i].pyplot(fig)
                else:
                    confusion_matrix_col2[0].pyplot(fig)

            st.markdown(f"### 📊 Overall {st.session_state.model_info['model_type']} Model Evaluation Results Data")
            st.dataframe(df)
