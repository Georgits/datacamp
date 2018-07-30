# 3. Chapter 3 ----
# 3.1 PCA using prcomp() ----
    # Perform scaled PCA: pr.out
    pr.out <- prcomp(iris[-5], scale = TRUE, center = TRUE)
    
    # Inspect model output
    summary(pr.out)
    
    # plot model output
    biplot(pr.out)
    
    # pront the pc-results
    pr.out$x
    
    
    # Variability of each principal component: pr.var
    pr.var <- pr.out$sdev^2
    
    # Variance explained by each principal component: pve
    pve <- pr.var / sum(pr.var)
    
    # Visualize variance explained
    # Plot variance explained for each principal component
    plot(pve, xlab = "Principal Component",
         ylab = "Proportion of Variance Explained",
         ylim = c(0, 1), type = "b")
    
    # Plot cumulative proportion of variance explained
    plot(cumsum(pve), xlab = "Principal Component",
         ylab = "Cummulative Proportion of Variance Explained",
         ylim = c(0, 1), type = "b")
    
    

    
# 3.2 Practical issues: scaling ----
    # PCA model with scaling: pr.with.scaling
    pr.with.scaling <- prcomp(iris[-5], scale = TRUE)
    
    # PCA model without scaling: pr.without.scaling
    pr.without.scaling <- prcomp(iris[-5], scale = FALSE)
    
    # Create biplots of both for comparison
    biplot(pr.with.scaling)
    biplot(pr.without.scaling)
    

    
# 4. Chapter 4 ----    
# 4.1 Preparing the data ----
    url <- "http://s3.amazonaws.com/assets.datacamp.com/production/course_1903/datasets/WisconsinCancer.csv"
    
    # Download the data: wisc.df
    wisc.df <- read.csv(url)
    
    # Convert the features of the data: wisc.data
    wisc.data <- as.matrix(wisc.df[,3:32])
    
    # Set the row names of wisc.data
    row.names(wisc.data) <- wisc.df$id
    
    # Create diagnosis vector
    diagnosis <- as.numeric(wisc.df$diagnosis == "M")
    
# 4.2 Performing PCA ----
    # Check column means and standard deviations
    colMeans(wisc.data)
    apply(wisc.data, 2, sd)
    
    # Execute PCA, scaling if appropriate: wisc.pr
    wisc.pr <- prcomp(wisc.data, scale = TRUE)
    
    # Look at summary of results
    summary(wisc.pr)
    
    
# 4.2 Interpreting PCA results ----
    # Create a biplot of wisc.pr
    biplot(wisc.pr)
    
    # Scatter plot observations by components 1 and 2
    plot(wisc.pr$x[, c(1, 2)], col = (diagnosis + 1), 
         xlab = "PC1", ylab = "PC2")
    
    # Repeat for components 1 and 3
    plot(wisc.pr$x[, c(1, 3)], col = (diagnosis + 1), 
         xlab = "PC1", ylab = "PC3")
    

# 4.3 Variance explained ----
    par(mfrow = c(1, 2))
    
    # Calculate variability of each component
    pr.var <- wisc.pr$sdev^2
    
    # Variance explained by each principal component: pve
    pve <- pr.var / sum(pr.var)
    
    # Plot variance explained for each principal component
    plot(pve, xlab = "Principal Component", 
         ylab = "Proportion of Variance Explained", 
         ylim = c(0, 1), type = "b")
    
    # Plot cumulative proportion of variance explained
    plot(cumsum(pve), xlab = "Principal Component", 
         ylab = "Cummulative Proportion of Variance Explained", 
         ylim = c(0, 1), type = "b")
    

# 4.4 Hierarchical clustering of case data ----
    # Scale the wisc.data data: data.scaled
    data.scaled <- scale(wisc.data)
    
    # Calculate the (Euclidean) distances: data.dist
    data.dist <- dist(data.scaled)
    
    # Create a hierarchical clustering model: wisc.hclust
    wisc.hclust <- hclust(data.dist, method = "complete")
    

# 4.5 Selecting number of clusters ----
    # Cut tree so that it has 4 clusters: wisc.hclust.clusters
    wisc.hclust.clusters <- cutree(wisc.hclust, h = 20)
    
    # Compare cluster membership to actual diagnoses
    table(wisc.hclust.clusters, diagnosis)

    
# 4.6 k-means clustering and comparing results ----
    # Create a k-means model on wisc.data: wisc.km
    wisc.km <- kmeans(scale(wisc.data), centers = 2, nstart = 20)
    
    # Compare k-means to actual diagnoses
    table(wisc.km$cluster, diagnosis)
    
    # Compare k-means to hierarchical clustering
    table(wisc.km$cluster, wisc.hclust.clusters)
    
    
# 4.7 Clustering on PCA results ----
    # Create a hierarchical clustering model: wisc.pr.hclust
    wisc.pr.hclust <- hclust(dist(wisc.pr$x[, 1:7]), method = "complete")
    
    # Cut model into 4 clusters: wisc.pr.hclust.clusters
    wisc.pr.hclust.clusters <- cutree(wisc.pr.hclust, k = 4)
    
    # Compare to actual diagnoses
    table(wisc.pr.hclust.clusters, diagnosis)
    
    # Compare to k-means and hierarchical
    table(wisc.km$cluster, diagnosis)
    table(wisc.hclust.clusters, diagnosis)