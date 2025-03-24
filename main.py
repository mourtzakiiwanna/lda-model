import os
from config import NUM_TOPICS, PASSES, RAW_DATA_DIR
from scripts.preprocess import load_and_preprocess_documents, extract_year_from_filename
from scripts.lda_modeling import train_lda_model, save_model_and_dictionary, visualize_topics
import pandas as pd
import matplotlib.pyplot as plt
from scripts.sentiment import train_skipgram_model, get_topic_sentiment

def run_for_country(country_name, path_to_speeches):
    print(f"\n=== Processing {country_name} ===")

    # Load & preprocess
    docs, filenames = load_and_preprocess_documents(path_to_speeches)
    if len(docs) < NUM_TOPICS:
        print(f"Skipping {country_name} due to too few documents.")
        return

    # Train LDA
    lda_model, corpus, dictionary = train_lda_model(docs, NUM_TOPICS, PASSES)

    # Create output folder
    out_folder = os.path.join("outputs", country_name)
    os.makedirs(out_folder, exist_ok=True)
    os.makedirs(os.path.join(out_folder, "visuals"), exist_ok=True)

    # Save model and visual
    save_model_and_dictionary(lda_model, dictionary, os.path.join(out_folder, "lda_model"))
    visualize_topics(lda_model, corpus, dictionary, os.path.join(out_folder, "visuals"))

    # Save topic-to-keywords summary
    keywords_summary = []

    for topic_id in range(NUM_TOPICS):
        top_words = lda_model.show_topic(topic_id, topn=10)
        word_list = ", ".join([word for word, _ in top_words])
        keywords_summary.append({
            "Topic": f"Topic_{topic_id}",
            "Top Words": word_list
        })

    df_keywords = pd.DataFrame(keywords_summary)
    df_keywords.to_excel(os.path.join(out_folder, "topics_keywords.xlsx"), index=False)
    print(f"Saved topic keyword summary to {os.path.join(out_folder, 'topics_keywords.xlsx')}")

    # Extract topic distributions
    topic_distributions = []
    years = []
    for doc, fname in zip(docs, filenames):
        bow = dictionary.doc2bow(doc)
        topics = lda_model.get_document_topics(bow, minimum_probability=0.0)
        topic_vector = [weight for _, weight in sorted(topics)]
        topic_distributions.append(topic_vector)
        years.append(extract_year_from_filename(fname))

    df = pd.DataFrame(topic_distributions, columns=[f"Topic_{i}" for i in range(NUM_TOPICS)])
    df["Year"] = years
    df = df.groupby("Year").mean().sort_index()
    df.to_csv(os.path.join(out_folder, "topics_by_year.csv"))
    print(f"Saved topics by year to {os.path.join(out_folder, 'topics_by_year.xlsx')}")

    # Plot
    df.plot(kind="line", marker='o')
    plt.title(f"Topic Distribution Over Years ({country_name})")
    plt.xlabel("Year")
    plt.ylabel("Topic Weight")
    plt.legend(loc="upper right")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(out_folder, "visuals", "topic_trends.png"))
    plt.close()
    print(f"Saved a plot with the topic distribution over years to {os.path.join(out_folder, 'visuals')} with name 'topic_trends.png'")

    # Build summary
    summary_rows = []

    for year, row in df.iterrows():
        top_topic_index = row.idxmax()
        top_topic_weight = row.max()
        topic_id = int(top_topic_index.split("_")[1])
        top_words = lda_model.show_topic(topic_id, topn=10)
        word_list = ", ".join([word for word, _ in top_words])

        summary_rows.append({
            "Year": year,
            "Top Topic": top_topic_index,
            "Weight": round(top_topic_weight, 2),
            "Top Words": word_list
        })

    # Save summary to Excel
    summary_df = pd.DataFrame(summary_rows)
    excel_path = os.path.join(out_folder, "top_topic_summary.xlsx")
    summary_df.to_excel(excel_path, index=False)
    print(f"Saved a summary of the top topic per year to {excel_path}")

    # Sentiment analysis
    # Step 1: Train Skip-gram model on tokenized docs
    skipgram_model = train_skipgram_model(docs)

    # Step 2: Sentiment analysis per topic
    sentiment_data = []
    for i in range(NUM_TOPICS):
        topic_words = [word for word, _ in lda_model.show_topic(i, topn=10)]
        sentiment, score = get_topic_sentiment(topic_words, skipgram_model)
        sentiment_data.append({
            "Topic": f"Topic_{i}",
            "Sentiment": sentiment,
            "Score": round(score, 3),
            "Top Words": ", ".join(topic_words)
        })

    # Step 3: Save to Excel
    sentiment_df = pd.DataFrame(sentiment_data)
    sentiment_path = os.path.join(out_folder, "topic_sentiments.xlsx")
    sentiment_df.to_excel(sentiment_path, index=False)
    print(f"Saved sentiment analysis per topic to {sentiment_path}")

def main():
    for folder_name in os.listdir(RAW_DATA_DIR):
        country_path = os.path.join(RAW_DATA_DIR, folder_name)
        if os.path.isdir(country_path):
            run_for_country(folder_name, country_path)

if __name__ == "__main__":
    main()
