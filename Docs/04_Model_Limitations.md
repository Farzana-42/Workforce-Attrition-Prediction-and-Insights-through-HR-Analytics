# Model Limitations & Ethical Considerations

## What This Model CANNOT Do

1. **Predict Personal Circumstances**
   - Cannot know about: illness, family relocation, spousal employment
   - Example: Employee with "low risk" leaves due to spouse's job transfer

2. **Capture Emotional State**
   - Only sees: ratings, ratings, output metrics
   - Cannot see: burnout, dissatisfaction, mental health
   - Example: High performer may be quietly job searching

3. **Predict Sudden Changes**
   - Trained on historical patterns
   - Cannot anticipate: market crashes, company crises, management changes
   - Example: New toxic manager → attrition spikes not in training data

4. **Account for Noise in Historical Data**
   - If past hiring/promotion was biased → model learns that bias
   - May perpetuate unfair patterns
   - Example: If women were historically rated lower, predictions may be skewed

5. **Be 100% Accurate**
   - 85.5% accuracy means 14.5% of predictions are wrong
   - Cannot determine which specific cases are wrong without ground truth

## Ethical Implications

### Risk of Misuse

1. **Not for Firing Decisions**
   - Never use this sole basis for termination
   - Would be discriminatory and unethical

2. **Not for Reduced Benefits**
   - Flagging someone as "at-risk" shouldn't reduce their compensation/benefits
   - Could incentivize actual departure

3. **Privacy Concerns**
   - Attrition predictions are sensitive personal data
   - Should be accessed only by authorized HR staff
   - Must comply with GDPR/CCPA regulations

4. **Self-Fulfilling Prophecy**
   - If flagged employees are treated differently, they may actually leave
   - Must use sensitively and supportively

### Bias Risks

Model may be biased if:
- Historical data underrepresented certain groups
- Past hiring/promotion decisions were discriminatory
- Certain cities/departments had systemic issues

**Mitigation**: Regularly audit predictions by demographic groups

## Recommended Usage

### DO

✅ Use as screening tool for proactive engagement
✅ Combine with manager judgment
✅ Have supportive conversations
✅ Track: Do flagged employees actually leave?
✅ Retrain model quarterly on new data
✅ Monitor performance by demographic group

### DON'T

❌ Use as sole basis for employment decisions
❌ Use to punish or demote employees
❌ Ignore predictions that contradict manager feedback
❌ Assume predictions are always correct
❌ Use without human oversight
❌ Share individual risk scores publicly

## Validation Strategy

### Monitor in Production

Track these metrics monthly:
- Recall: Of people who actually left, how many did we flag?
- Precision: Of people we flagged, how many actually left?
- Drift: Is model accuracy declining over time?

### Retraining Schedule

- Retrain quarterly with new attrition data
- If accuracy drops below 80%, investigate causes
- If data distribution changes significantly, consider model rebuild

## Conclusion

This model is a powerful tool for HR decision support, but NOT a replacement for human judgment. Use it to start conversations, not end them.