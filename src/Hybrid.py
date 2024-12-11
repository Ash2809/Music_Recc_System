def hybrid_filter(dataset,n,track_index,weight=0.7):
    content_scores=content_filter(dataset,n,track_index)
    collab_recs=collaborative_filter(dataset,n)

    max_content=max(content_scores.values())
    normalized_content={k:v/max_content for k,v in content_scores.items()}

    max_collab=collab_recs["collab_score"].max()
    collab_recs["normalized_collab"]=collab_recs["collab_score"]/max_collab

    hybrid_score={}
    for idx in content_scores:
        content_score=normalized_content.get(idx,0)
        collab_score=collab_recs[collab_recs.index==idx]["normalized_collab"].sum()
        hybrid_score['idx']= weight*content_score + (1-weight)*collab_score
    
    sort_hybrid=sorted(hybrid_score.items())

    return sort_hybrid