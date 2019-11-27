function result = Entropy(h)
    TotalNum = length(h.Data);
    Probs = h.BinCounts/TotalNum;
    result = 0;
    for r = 1:length(Probs)
        temp = Probs(r,:);
        temp = temp(temp~=0);
        result = result - sum(temp.*log(temp));
    end
end

