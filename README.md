# Aspect-Based Sentiment Analysis for Open-Ended Survey responses
Code for thesis project Information Studies: Data Science, University of Amsterdam (2022-2023) <br />
Student: Lois Rink <br />
Thesis project completed at: Randstad Groep Nederland <br />


## Abstract
Open-ended survey responses, with their diverse writing styles and topics, serve as a valuable source of information. However, quantifying this information —especially for sentiment analysis —poses a challenge. Traditional approaches struggle to capture nuanced sentiments, potentially overlooking critical aspects that trigger positive or negative sentiments. This research addresses this problem by introducing a machine learning approach for aspect-based sentiment analysis (ABSA) of Dutch open-ended responses in employee satisfaction surveys. This approach aims to overcome the inherent noise and variability in these responses, enabling a more comprehensive analysis of sentiments. K-means clustering identifies six key aspects (salary, schedule, contact, communication, personal attention, agreements) that are validated by domain experts. A data set of 1458 Dutch responses is compiled, revealing label imbalance in aspects and sentiments. Two few-shot machine learning approaches for ABSA —BERTje (aspect: F1 = 0.52, sentiment: F1 = 0.87) and RobBERT (aspect: F1 = 0.61, sentiment: F1 = 0.89) —are
developed and compared against SVM, MLP and zero-shot baselines. The approach offers an effective solution for automated processing of open-ended survey responses, translating unstructured data into actionable insights. This work significantly contributes to the field of ABSA by demonstrating the first successful application of Dutch pre-trained language models to aspect-specific sentiment analysis in the domain of human resources (HR).

## Note
To protect the privacy of the respondents who filled out the surveys, the data for this research will not be published.
