library(qdap)
library(tm)
library(RWeka)
library(wordcloud)
# setting working directory
setwd("C:\\Users\\Sdhiraj\\Documents\\My_R_space\\datasets\\Amazon vs Google")
# reading data
amzn <- read.csv("amzn.csv", stringsAsFactors = FALSE)
goog <- read.csv("gogl.csv", stringsAsFactors = FALSE)

str(amzn)
amzn_pros <- amzn$pros
amzn_cons <- amzn$cons

str(gogl)
goog_pros <- goog$pros
goog_cons <- goog$cons

# qdap cleaning function which cleans a text vector
qdap_clean <- function(x){
              x <- replace_abbreviation(x)
              x <- replace_contraction(x)
              x <- replace_number(x)
              x <- replace_ordinal(x)
              x <- replace_symbol(x)
              x <- tolower(x)
              return(x)
              }
# clean amzn text vector
amzn_pros <- qdap_clean(amzn_pros)
amzn_cons <- qdap_clean(amzn_cons)
# clean goog text vector
goog_pros <- qdap_clean(goog_pros)
goog_cons <- qdap_clean(goog_cons)

# tm cleaning function which cleans a corpus object
tm_clean <- function(corpus){
            corpus <- tm_map(corpus, removePunctuation)
            corpus <- tm_map(corpus, stripWhitespace)
            corpus <- tm_map(corpus, removeWords, 
                             c(stopwords("en"), "Google", "Amazon", "company"))
            return(corpus)     
            }
# creating a VCorpus object for amzn reviews
az_p_corp <- VCorpus(VectorSource(amzn_pros))
az_c_corp <- VCorpus(VectorSource(amzn_cons))

# creating a VCorpus object for goog reviews
goog_p_corp <- VCorpus(VectorSource(goog_pros))
goog_c_corp <- VCorpus(VectorSource(goog_cons))

# applying tm_clean function to corpus
amzn_pros_corp <- tm_clean(az_p_corp)
amzn_cons_corp <- tm_clean(az_c_corp)
goog_pros_corp <- tm_clean(goog_p_corp)
goog_cons_corp <- tm_clean(goog_c_corp)


# Feature extraction & analysis
tokenizer <- function(x) {
  NGramTokenizer(x, Weka_control(min = 2, max = 2))
}


# amazon feature extraction using bi-grams
# amzn_pros feature extraction 
amzn_p_tdm <- TermDocumentMatrix(amzn_pros_corp, control = list(tokenize = tokenizer))
amzn_p_tdm_m <- as.matrix(amzn_p_tdm)
amzn_p_freq <- rowSums(amzn_p_tdm_m)

# Plot a wordcloud using amzn_p_freq values
wordcloud(names(amzn_p_freq), amzn_p_freq, max.words = 25, color = "blue")

# amzn_cons feature extraction
amzn_c_tdm <- TermDocumentMatrix(amzn_cons_corp, control = list(tokenize = tokenizer))
amzn_c_tdm_m <- as.matrix(amzn_c_tdm)
amzn_c_freq <- rowSums(amzn_c_tdm_m)

# Plot a wordcloud of negative Amazon bigrams
wordcloud(names(amzn_c_freq), amzn_c_freq, max.words = 25, color = "red")

# google feature extraction using bi-grams
# google pros word cloud
goog_p_tdm <- TermDocumentMatrix(goog_pros_corp, control = list(tokenize = tokenizer))
goog_p_tdm_m <- as.matrix(goog_p_tdm)
goog_p_freq <- rowSums(goog_p_tdm_m)

# Plot a wordcloud of negative google bigrams
wordcloud(names(goog_p_freq), goog_p_freq, max.words = 25, color = "red")

# google cons word cloud
goog_c_tdm <- TermDocumentMatrix(goog_cons_corp, control = list(tokenize = tokenizer))
goog_c_tdm_m <- as.matrix(goog_c_tdm)
goog_c_freq <- rowSums(goog_c_tdm_m)

# Plot a wordcloud of negative google bigrams
wordcloud(names(goog_c_freq), goog_c_freq, max.words = 25, color = "red")




# google review by comparison cloud and commonality cloud
# technique - 1 using bi-grams
goog_pros_1 <- paste(goog_pros, collapse = " ")
goog_cons_1 <- paste(goog_cons, collapse = " ")
all_goog <- c(goog_pros_1, goog_cons_1)
all_goo_corpus <- VCorpus(VectorSource(all_goog))
all_goo_corp <- tm_clean(all_goo_corpus)

all_tdm <- TermDocumentMatrix(all_goo_corp, control = list(tokenize = tokenizer))
all_m <- as.matrix(all_tdm)
colnames(all_m) <- c("Google_pros", "Google_cons")
comparison.cloud(all_m, random.order = FALSE, colors = c("#F44336", "#2196f3"),title.size=1.5, max.words = 100)
commonality.cloud(all_m, random.order = FALSE, colors = brewer.pal(8, "Dark2"),title.size=1.5)

# technique - 2 using uni-grams
all_tdm_uni_goog <- TermDocumentMatrix(all_goo_corp)
all_m_uni_goog <- as.matrix(all_tdm_uni_goog)
colnames(all_m_uni_goog) <- c("Google_pros", "Google_cons")
comparison.cloud(all_m_uni_goog, random.order = FALSE, colors = c("#F44336", "#2196f3"),title.size=1.5, max.words = 100)
commonality.cloud(all_m_uni_goog, random.order = FALSE, colors = brewer.pal(8, "Dark2"),title.size=1.5)

# amazon review by comparison cloud and commonality cloud
# technique - 1 using bi-grams
amzn_pros_1 <- paste(amzn_pros, collapse = " ")
amzn_cons_1 <- paste(amzn_cons, collapse = " ")
all_amzn <- c(amzn_pros_1, amzn_cons_1)
all_amzn_corpus <- VCorpus(VectorSource(all_amzn))
all_amzn_corp <- tm_clean(all_amzn_corpus)

all_tdm_bi_amzn <- TermDocumentMatrix(all_amzn_corp, control = list(tokenize = tokenizer))
all_m_bi_amzn <- as.matrix(all_tdm_bi_amzn)
colnames(all_m_bi_amzn) <- c("Amazon_pros", "Amazon_cons")
comparison.cloud(all_m_bi_amzn, random.order = FALSE, colors = c("#F44336", "#2196f3"),title.size=1.5, max.words = 100)
commonality.cloud(all_m_bi_amzn, random.order = FALSE, colors = brewer.pal(8, "Dark2"),title.size=1.5)

# technique - 2 using uni-grams
all_tdm_uni_amzn <- TermDocumentMatrix(all_amzn_corp)
all_m_uni_amzn <- as.matrix(all_tdm_uni_amzn)
colnames(all_m_uni_amzn) <- c("Amazon_pros", "Amazon_cons")
comparison.cloud(all_m_uni_amzn, random.order = FALSE, colors = c("#F44336", "#2196f3"),title.size=1.5, max.words = 100)
commonality.cloud(all_m_uni_amzn, random.order = FALSE, colors = brewer.pal(8, "Dark2"),title.size=1.5)







