
## Evaluation

### Performance:
- learning curves and r-squared curves
- plot ground truth xy coordinates with small scatter plots of predicted coordinates

### Inner workings:
- mean activity, coded by target location vs target direction, then separately for all four io types
- PCA at each timepoint, then plot timeseries of principal components, colour coded by start/ goal location and direction
- "RSA": marginalise irrelevant dimensions, one RDM per start loc, goal loc, goal direction
- RSA: model rdms based on xy target locations and direction (angles) the latter with cosine dissimilarity