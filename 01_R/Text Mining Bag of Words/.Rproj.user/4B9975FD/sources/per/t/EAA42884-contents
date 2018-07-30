# Chapter 1.  Jumping into text mining with bag of words ----
# Quick taste of text mining ----

new_text <- "DataCamp is the first online learning platform that focuses on building the best learning experience specifically for Data Science. We have offices in Boston and Belgium and to date, we trained over 250,000 (aspiring) data scientists in over 150 countries. These data science enthusiasts completed more than 9 million exercises. You can take free beginner courses, or subscribe for $25/month to get access to all premium courses."

# Load qdap
library(qdap)

# Print new_text to the console
print(new_text)

# Find the 10 most frequent terms: term_count
term_count <- freq_terms(new_text, 10)

# Plot term_count
plot(term_count)




# Load some text -----
# Import text data
tweets <- read.csv("coffee.csv", stringsAsFactors = FALSE)

# View the structure of tweets
str(tweets)

# Print out the number of rows in tweets
nrow(tweets)

# Isolate text from tweets: coffee_tweets
coffee_tweets <- tweets$text




# Make the vector a VCorpus object (1) ----
# Load tm
library(tm)

# Make a vector source: coffee_source
coffee_source <- VectorSource(coffee_tweets)




# Make the vector a VCorpus object (2) ----
## coffee_source is already in your workspace

# Make a volatile corpus: coffee_corpus
coffee_corpus <- VCorpus(coffee_source)

# Print out coffee_corpus
coffee_corpus

# Print data on the 15th tweet in coffee_corpus
coffee_corpus[[15]]

# Print the content of the 15th tweet in coffee_corpus
coffee_corpus[[15]][1]



# Make a VCorpus from a data frame ----
# Print example_text to the console
example_text

# Create a DataframeSource on columns 2 and 3: df_source
df_source <- DataframeSource(example_text[, 2:3])

# Convert df_source to a corpus: df_corpus
df_corpus <- VCorpus(df_source)

# Examine df_corpus
df_corpus

# Create a VectorSource on column 3: vec_source
vec_source <- VectorSource(example_text[,3])

# Convert vec_source to a corpus: vec_corpus
vec_corpus <-  VCorpus(vec_source)

# Examine vec_corpus
vec_corpus



# Common cleaning functions from tm ----
# Create the object: text
text <- "<b>She</b> woke up at       6 A.M. It\'s so early!  She was only 10% awake and began drinking coffee in front of her computer."

# All lowercase
tolower(text)

# Remove punctuation
removePunctuation(text)

# Remove numbers
removeNumbers(text)

# Remove whitespace
stripWhitespace(text)




# Cleaning with qdap ----
## text is still loaded in your workspace

# Remove text within brackets
bracketX(text)

# Replace numbers with words
replace_number(text)

# Replace abbreviations
replace_abbreviation(text)

# Replace contractions
replace_contraction(text)

# Replace symbols with words
replace_symbol(text)



# All about stop words ----
## text is preloaded into your workspace
# List standard English stop words
stopwords("en")

# Print text without standard stop words
removeWords(text, stopwords("en"))

# Add "coffee" and "bean" to the list: new_stops
new_stops <- c("coffee", "bean", stopwords("en"))

# Remove stop words from text
removeWords(text, new_stops)



# Intro to word stemming and stem completion ----
# Create complicate
complicate <- c("complicated", "complication", "complicatedly")

# Perform word stemming: stem_doc
stem_doc <- stemDocument(complicate)

# Create the completion dictionary: comp_dict
comp_dict <- c("complicate")

# Perform stem completion: complete_text 
complete_text <- stemCompletion(stem_doc, comp_dict)

# Print complete_text
complete_text



# Word stemming and stem completion on a sentence ----
text_data <- "In a complicated haste, Tom rushed to fix a new complication, too complicatedly."
comp_dict <- c("In", "a", "complicate", "haste", "Tom", "rush", "to", "fix", "new", "too")

# Remove punctuation: rm_punc
rm_punc <- removePunctuation(text_data)

# Create character vector: n_char_vec
n_char_vec <- unlist(strsplit(rm_punc, split = ' '))

# Perform word stemming: stem_doc
stem_doc <- stemDocument(n_char_vec)

# Print stem_doc
stem_doc

# Re-complete stemmed document: complete_doc
complete_doc <- stemCompletion(stem_doc, comp_dict)

# Print complete_doc
complete_doc




# Apply preprocessing steps to a corpus ----
# Notice how the tm package functions do not need content_transformer(), but base R and qdap functions do.
# Alter the function code to match the instructions
clean_corpus <- function(corpus){
  corpus <- tm_map(corpus, stripWhitespace)
  corpus <- tm_map(corpus, removePunctuation)
  corpus <- tm_map(corpus, content_transformer(tolower))
  corpus <- tm_map(corpus, removeWords, c(stopwords("en"), "coffee"))
  corpus <- tm_map(corpus, content_transformer(replace_abbreviation))
  corpus <- tm_map(corpus, removeNumbers)
  return(corpus)
}

# Apply your customized function to the tweet_corp: clean_corp
clean_corp <- clean_corpus(tweet_corp)

# Print out a cleaned up tweet
clean_corp[[227]][1]

# Print out the same tweet in original form
tweet_corp[[227]][1]




# Make a document-term matrix ----
# Create the dtm from the corpus: coffee_dtm
coffee_dtm <- DocumentTermMatrix(clean_corp)

# Print out coffee_dtm data
coffee_dtm

# Convert coffee_dtm to a matrix: coffee_m
coffee_m <- as.matrix(coffee_dtm)

# Print the dimensions of coffee_m
dim(coffee_m)

# Review a portion of the matrix
coffee_m[148:150,2587:2590]



# Make a term-document matrix ----
# Create a TDM from clean_corp: coffee_tdm
coffee_tdm <- TermDocumentMatrix(clean_corp)

# Print coffee_tdm data
coffee_tdm

# Convert coffee_tdm to a matrix: coffee_m
coffee_m <- as.matrix(coffee_tdm)

# Print the dimensions of the matrix
dim(coffee_m)

# Review a portion of the matrix
coffee_m[2587:2590, 148:150]



# Chapter 2. Word clouds and more interesting visuals ----
# Frequent terms with tm ----
## coffee_tdm is still loaded in your workspace

# Create a matrix: coffee_m
coffee_m <- as.matrix(coffee_tdm)

# Calculate the rowSums: term_frequency
term_frequency <- rowSums(coffee_m)

# Sort term_frequency in descending order
term_frequency <- sort(term_frequency, decreasing = TRUE)

# View the top 10 most common words
term_frequency[1:10]

# Plot a barchart of the 10 most common words
barplot(term_frequency[1:10], col = "tan", las = 2)



# Frequent terms with qdap ---
# Create frequency
frequency <- freq_terms(tweets$text, top = 10, at.least = 3, stopwords = "Top200Words")

# Make a frequency barchart
plot(frequency)

# Create frequency2
frequency2 <- freq_terms(tweets$text, top = 10, at.least = 3, tm::stopwords("english"))

# Make a frequency2 barchart
plot(frequency2)




# A simple word cloud ----
## term_frequency is loaded into your workspace

# Load wordcloud package
library(wordcloud)

# Print the first 10 entries in term_frequency[1:10]
term_frequency[1:10]

# Create word_freqs
word_freqs <- data.frame(term = names(term_frequency), num = term_frequency)

# Create a wordcloud for the values in word_freqs
wordcloud(word_freqs$term, word_freqs$num, max.words = 100, colors = "red")



# Stop words and word clouds ----
# Add new stop words to clean_corpus()
clean_corpus <- function(corpus){
  corpus <- tm_map(corpus, removePunctuation)
  corpus <- tm_map(corpus, stripWhitespace)
  corpus <- tm_map(corpus, removeNumbers)
  corpus <- tm_map(corpus, content_transformer(tolower))
  corpus <- tm_map(corpus, removeWords, 
                   c(stopwords("en"), "amp", "chardonnay", "wine", "glass"))
  return(corpus)
}

# Create clean_chardonnay
clean_chardonnay <- clean_corpus(chardonnay_corp)

# Create chardonnay_tdm
chardonnay_tdm <- TermDocumentMatrix(clean_chardonnay)

# Create chardonnay_m
chardonnay_m <- as.matrix(chardonnay_tdm)

# Create chardonnay_words
chardonnay_words <- rowSums(chardonnay_m)



# Plot the better word cloud ----
# Sort the chardonnay_words in descending order
chardonnay_words <- sort(chardonnay_words, decreasing = TRUE)

# Print the 6 most frequent chardonnay terms
chardonnay_words[1:6]

# Create chardonnay_freqs
chardonnay_freqs <- data.frame(term = names(chardonnay_words), num = chardonnay_words)

# Create a wordcloud for the values in word_freqs
wordcloud(chardonnay_freqs$term, chardonnay_freqs$num, max.words = 50, color = "red")




# Improve word cloud colors ----
# Print the list of colors
colors()

# Print the wordcloud with the specified colors
wordcloud(chardonnay_freqs$term, chardonnay_freqs$num, max.words = 100, color = c("grey80", "darkgoldenrod1", "tomato"))



# Use prebuilt color palettes ----
# List the available colors
display.brewer.all()

# Create purple_orange
purple_orange <- brewer.pal(10, "PuOr")

# Drop 2 faintest colors
purple_orange <- purple_orange[-(1:2)]

# Create a wordcloud with purple_orange palette
wordcloud(chardonnay_freqs$term, chardonnay_freqs$num, max.words = 100, color = purple_orange)




# Find common words ----
# Create all_coffee
all_coffee <- paste(coffee_tweets$text, collapse = " ")

# Create all_chardonnay
all_chardonnay <- paste(chardonnay_tweets$text, collapse = " ")

# Create all_tweets
all_tweets <- c(all_coffee, all_chardonnay)

# Convert to a vector source
all_tweets <- VectorSource(all_tweets)

# Create all_corpus
all_corpus <- VCorpus(all_tweets)





# Visualize common words ----
# Clean the corpus
all_clean <- clean_corpus(all_corpus)

# Create all_tdm
all_tdm <- TermDocumentMatrix(all_clean)

# Create all_m
all_m <- as.matrix(all_tdm)

# Print a commonality cloud
commonality.cloud(all_m, max.words = 100, colors = "steelblue1")



# Visualize dissimilar words ----
# Clean the corpus
all_clean <- clean_corpus(all_corpus)

# Create all_tdm
all_tdm <- TermDocumentMatrix(all_clean)

# Give the columns distinct names
colnames(all_tdm) <- c("coffee", "chardonnay")

# Create all_m
all_m <- as.matrix(all_tdm)

# Create comparison cloud
comparison.cloud(all_m, colors = c("orange", "blue"), max.words = 50)




# Polarized tag cloud ----
all_tdm_m <- as.matrix(all_tdm)

# Create common_words
common_words <- subset(all_tdm_m, all_tdm_m[, 1] > 0 & all_tdm_m[, 2] > 0)

# Create difference
difference <- abs(common_words[, 1] - common_words[, 2])

# Combine common_words and difference
common_words <- cbind(common_words, difference)

# Order the data frame from most differences to least
common_words <- common_words[order(common_words[, 3], decreasing = TRUE), ]

# Create top25_df
top25_df <- data.frame(x = common_words[1:25, 1], 
                       y = common_words[1:25, 2], 
                       labels = rownames(common_words[1:25, ]))

# Create the pyramid plot
pyramid.plot(top25_df$x, top25_df$y,
             labels = top25_df$labels, gap = 8,
             top.labels = c("Chardonnay", "Words", "Coffee"),
             main = "Words in Common", laxlab = NULL, 
             raxlab = NULL, unit = NULL)





# Visualize word networks ----
# Word association
word_associate(coffee_tweets$text, match.string = c("barista"), 
               stopwords = c(Top200Words, "coffee", "amp"), 
               network.plot = TRUE, cloud.colors = c("gray85", "darkred"))

# Add title
title(main = "Barista Coffee Tweet Associations")




# Chapter 3. Adding to your tm skills ----
# Distance matrix and dendrogram ----
# Create dist_rain
dist_rain <- dist(rain[,2])

# View the distance matrix
dist_rain

# Create hc
hc <- hclust(dist_rain)

# Plot hc
plot(hc, labels = rain$city)




# Make a distance matrix and dendrogram from a TDM ----
# Print the dimensions of tweets_tdm
dim(tweets_tdm)

# Create tdm1
tdm1 <- removeSparseTerms(tweets_tdm, sparse = 0.95)

# Create tdm2
tdm2 <- removeSparseTerms(tweets_tdm, sparse = 0.975)

# Print tdm1
tdm1

# Print tdm2
tdm2



# Put it all together: a text based dendrogram ----
# Create tweets_tdm2
tweets_tdm2 <- removeSparseTerms(tweets_tdm, sparse = 0.975)

# Create tdm_m
tdm_m <- as.matrix(tweets_tdm2)

# Create tdm_df
tdm_df <- as.data.frame(tdm_m)

# Create tweets_dist
tweets_dist <- dist(tdm_df)

# Create hc
hc <- hclust(tweets_dist)

# Plot the dendrogram
plot(hc)




# Dendrogram aesthetics ----
# Load dendextend
library(dendextend)

# Create hc
hc <- hclust(tweets_dist)

# Create hcd
hcd <- as.dendrogram(hc)

# Print the labels in hcd
labels(hcd)

# Change the branch color to red for "marvin" and "gaye"
hcd <- branches_attr_by_labels(hcd, c("marvin", "gaye"), color = "red")

# Plot hcd
plot(hcd, main = "Better Dendrogram")

# Add cluster rectangles 
rect.dendrogram(hcd, k = 2, border = "grey50")





# Using word association ----
# Create associations
associations <- findAssocs(tweets_tdm, "venti", 0.2)

# View the venti associations
associations

# Create associations_df
associations_df <- list_vect2df(associations)[, 2:3]

# Plot the associations_df values (don't change this)
ggplot(associations_df, aes(y = associations_df[, 1])) + 
  geom_point(aes(x = associations_df[, 2]), 
             data = associations_df, size = 3) + 
  theme_gdocs()



# Changing n-grams ----
# Make tokenizer function 
tokenizer <- function(x) 
  NGramTokenizer(x, Weka_control(min = 2, max = 2))

# Create unigram_dtm
unigram_dtm <- DocumentTermMatrix(text_corp)

# Create bigram_dtm
bigram_dtm <- DocumentTermMatrix(text_corp, control = list(tokenize = tokenizer))

# Examine unigram_dtm
unigram_dtm

# Examine bigram_dtm
bigram_dtm



# How do bigrams affect word clouds? ----
# Create bigram_dtm_m
bigram_dtm_m <- as.matrix(bigram_dtm)

# Create freq
freq <- colSums(bigram_dtm_m)

# Create bi_words
bi_words <- names(freq)

# Examine part of bi_words
bi_words[2577:2587]

# Plot a wordcloud
wordcloud(bi_words, freq = freq, max.words = 15)



# Changing frequency weights ----
# Create tf_tdm
tf_tdm <- TermDocumentMatrix(text_corp)

# Create tfidf_tdm
tfidf_tdm <- TermDocumentMatrix(text_corp, control = list(weighting = weightTfIdf))

# Create tf_tdm_m
tf_tdm_m <- as.matrix(tf_tdm)

# Create tfidf_tdm_m 
tfidf_tdm_m <- as.matrix(tfidf_tdm)

# Examine part of tf_tdm_m
tf_tdm_m[508:509,5:10]

# Examine part of tfidf_tdm_m
tfidf_tdm_m[508:509,5:10]





# Capturing metadata in tm ----
# Add author to custom reading list
custom_reader <- readTabular(mapping = list(content = "text", 
                                            id = "num",
                                            author = "screenName",
                                            date = "created"
))

# Make corpus with custom reading
text_corpus <- VCorpus(
  DataframeSource(tweets), 
  readerControl = list(reader = custom_reader)
)

# Clean corpus
text_corpus <- clean_corpus(text_corpus)

# Print data
text_corpus[[1]][1]

# Print metadata
text_corpus[[1]][2]




# Chapter 4. Battle of the tech giants for talent ----
# Step 1: Problem definition----
# Step 2: Identifying the text sources ----
# Print the structure of amzn
str(amzn)

# Create amzn_pros
amzn_pros <- amzn$pros

# Create amzn_cons
amzn_cons <- amzn$cons

# Print the structure of goog
str(goog)

# Create goog_pros
goog_pros <- goog$pros

# Create goog_cons
goog_cons <- goog$cons



# Text organization ----

# applies a series of qdap functions to a text vector
qdap_clean <- function(x){
  x <- replace_abbreviation(x)
  x <- replace_contraction(x)
  x <- replace_number(x)
  x <- replace_ordinal(x)
  x <- replace_ordinal(x)
  x <- replace_symbol(x)
  x <- tolower(x)
  return(x)
}


# applies a series of tm functions to a corpus object
tm_clean <- function(corpus){
  corpus <- tm_map(corpus, removePunctuation)
  corpus <- tm_map(corpus, stripWhitespace)
  corpus <- tm_map(corpus, removeWords, 
                   c(stopwords("en"), "Google", "Amazon", "company"))
  return(corpus)
}

# Alter amzn_pros
amzn_pros <- qdap_clean(amzn_pros)

# Alter amzn_cons
amzn_cons <- qdap_clean(amzn_cons)

# Create az_p_corp 
az_p_corp <- VCorpus(VectorSource(amzn_pros))

# Create az_c_corp
az_c_corp <- VCorpus(VectorSource(amzn_cons))

# Create amzn_pros_corp
amzn_pros_corp <- tm_clean(az_p_corp)

# Create amzn_cons_corp
amzn_cons_corp <- tm_clean(az_c_corp)





# Working with Google reviews ----
# Apply qdap_clean to goog_pros
goog_pros <- qdap_clean(goog_pros)

# Apply qdap_clean to goog_cons
goog_cons <- qdap_clean(goog_cons)

# Create goog_p_corp
goog_p_corp <- VCorpus(VectorSource(goog_pros))

# Create goog_c_corp
goog_c_corp <- VCorpus(VectorSource(goog_cons))

# Create goog_pros_corp
goog_pros_corp <- tm_clean(goog_p_corp)

# Create goog_cons_corp
goog_cons_corp <- tm_clean(goog_c_corp)





# Steps 4 & 5: Feature extraction & analysis ----
# Feature extraction & analysis: amzn_pros ----
library(Rweka)
tokenizer <- function(x) 
  NGramTokenizer(x, Weka_control(min = 2, max = 2))

# Create amzn_p_tdm
amzn_p_tdm <- TermDocumentMatrix(amzn_pros_corp, list(tokenize = tokenizer))

# Create amzn_p_tdm_m
amzn_p_tdm_m <- as.matrix(amzn_p_tdm)

# Create amzn_p_freq
amzn_p_freq <- rowSums(amzn_p_tdm_m)

# Plot a wordcloud using amzn_p_freq values
wordcloud(names(amzn_p_freq), max.words = 25, color = "blue")




# Feature extraction & analysis: amzn_cons ----
# Create amzn_c_tdm
amzn_c_tdm <- TermDocumentMatrix(amzn_cons_corp, list(tokenize = tokenizer))

# Create amzn_c_tdm_m
amzn_c_tdm_m <- as.matrix(amzn_c_tdm)

# Create amzn_c_freq
amzn_c_freq <- rowSums(amzn_c_tdm_m)

# Plot a wordcloud of negative Amazon bigrams
wordcloud(names(amzn_c_freq), max.words = 25, color = "blue")




# amzn_cons dendrogram ----
# Create amzn_c_tdm
amzn_c_tdm <- TermDocumentMatrix(amzn_cons_corp, control = list(tokenize = tokenizer))

# Print amzn_c_tdm to the console
amzn_c_tdm

# Create amzn_c_tdm2 by removing sparse terms 
amzn_c_tdm2 <- removeSparseTerms(amzn_c_tdm, sparse = 0.993) 

# Create hc as a cluster of distance values
hc <- hclust(dist(amzn_c_tdm2, method = "euclidean"), method = "complete")

# Produce a plot of hc
plot(hc)



# Word association ----
# Create amzn_p_tdm
amzn_p_tdm <- TermDocumentMatrix(amzn_pros_corp, control = list(tokenize = tokenizer))

# Create amzn_p_m
amzn_p_m <- as.matrix(amzn_p_tdm)

# Create amzn_p_freq
amzn_p_freq <- rowSums(amzn_p_m)

# Create term_frequency
term_frequency <- sort(amzn_p_freq, decreasing = TRUE)

# Print the 5 most common terms
term_frequency[1:5]

# Find associations with fast paced
findAssocs(amzn_p_tdm, "fast paced", 0.2)




# Quick review of Google reviews ----
# We've provided you with a corpus all_goog_corpus, which has the 500 positive and 500 negative reviews for Google. 
# Wie dies geschieht s. oben, bei all_tweets

# Create all_goog_corp
all_goog_corp <- tm_clean(all_goog_corpus)

# Create all_tdm
all_tdm <- TermDocumentMatrix(all_goog_corp)

# Name the columns of all_tdm
colnames(all_tdm) <- c("Goog_Pros", "Goog_Cons")

# Create all_m
all_m <- as.matrix(all_tdm)

# Build a comparison cloud
comparison.cloud(all_m, max.word = 100, colors = c("#F44336", "#2196f3"))





# Cage match! Amazon vs. Google pro reviews ----
# Create common_words
common_words <- subset(all_tdm_m, all_tdm_m[, 1] > 0 & all_tdm_m[, 2] > 0)

# Create difference
difference <- abs(common_words[, 1] - common_words[, 2])

# Add difference to common_words
  common_words <- cbind(common_words, difference)

# Order the data frame from most differences to least
common_words <- common_words[order(common_words[, 3], decreasing = TRUE), ]

# Create top15_df
top15_df <- data.frame(
  x = common_words[1:15, 1],
  y = common_words[1:15, 2],
  labels = rownames(common_words[1:15, ]))

# Create the pyramid plot
pyramid.plot(top15_df$x, top15_df$y, 
             labels = top15_df$labels, gap = 12, 
             top.labels = c("Amzn", "Pro Words", "Google"), 
             main = "Words in Common", unit = NULL)






# Cage match, part 2! Negative reviews ----
# all_goog_corpus muss nun mit cons (negativen)-begriffsüpaaren gebildet werden.
# Create common_words
common_words <- subset(all_tdm_m, all_tdm_m[, 1] > 0 & all_tdm_m[, 2] > 0)

# Create difference
difference <- abs(common_words[, 1] - common_words[, 2])

# Bind difference to common_words
common_words <- cbind(common_words, difference)

# Order the data frame from most differences to least
common_words <- common_words[order(common_words[, 3], decreasing = TRUE), ]

# Create top15_df
top15_df <- data.frame(
  x = common_words[1:15, 1],
  y = common_words[1:15, 2],
  labels = rownames(common_words[1:15, ]))

# Create the pyramid plot
pyramid.plot(top15_df$x, top15_df$y, 
             labels = top15_df$labels, gap = 12, 
             top.labels = c("Amzn", "Cons Words", "Google"), 
             main = "Words in Common", unit = NULL)
