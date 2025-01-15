# Adjustment of hyperparameters (loss weights):

If you intend to apply our proposed LCA mechanism to different tasks, please refer to the following principles for adjusting the loss function weights:

1. The loss weights are essentially hyperparameters adjusted based on empirical knowledge, but their configuration must align with the characteristics of the proposed LCA mechanism. Our learning process primarily focuses on the main branch; therefore, when setting the cross-entropy loss for the classification tasks of the main branch and the counterfactual attention branch, the weight ($\lambda_{ce}^{main}$) assigned to the main branch should be greater than that assigned to the counterfactual attention branch ($\lambda_{ce}^{cf}$), as the latter serves as a supportive mechanism to enhance the overall learning process. 

2. Building on the concept outlined above, to guide the model toward focusing on biased regions and mitigating overfitting effects via entropy loss, the entropy loss weight for the counterfactual attention branch ($\lambda_{ent}^{cf}$) should be set higher than that for the main branch ($\lambda_{ent}^{main}$). 

3. We treat the loss weight $\lambda_{ce}^{effect}$, which controls the evaluation of classification performance between the main branch and the counterfactual attention branch and encourages the main branch to focus on more discriminative regions, as the baseline value, setting it to 1 (for all tasks) to guide the adjustment of other weights. 

4. Finally, the loss weight $\lambda_{1}^{att}$, which controls the difference between the attention maps of the main branch and the counterfactual attention branch, is adjusted based on the weights established above. 

By adhering to the principles outlined above, these weights (hyperparameters) can be effectively configured and further fine-tuned through a small-scale grid search to achieve desirable results across various tasks.