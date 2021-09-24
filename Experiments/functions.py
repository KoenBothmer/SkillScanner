def get_cluster_top_n(cluster, n):
    result_ids = []
    result_scores = []
    result_requirements = []
    embeddings_f = embeddings_test.astype(float)
    top_k = min(n, len(embeddings_test))
    
    cos_scores = util.pytorch_cos_sim(kmeans.cluster_centers_[cluster], embeddings_f)[0]
    top_results = torch.topk(cos_scores, k=top_k)
    for score, idx in zip(top_results[0], top_results[1]):
        #print(df_r.at[idx.item(),'requirement']+" Score: "+str(score.item()))
        result_ids.append(idx.item())
        result_requirements.append(df_r_test.at[idx.item(),'requirement'])
        result_scores.append(score.item())
    return([result_ids[:n],result_requirements[:n],result_scores[:n]])
get_cluster_top_n(20,3)


def get_bigrams(requirement_list):
    from collections import Counter
    list = []
    for requirement in requirement_list[1]:
        for word in requirement.split():
            word = word.lower()
            if (len(word) >1) & (word not in stop_words):
                list.append(word)
    
    bigrams = zip(list,list[1:])
    bigrams_count = Counter(bigrams)
    return(bigrams_count)