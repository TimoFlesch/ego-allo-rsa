
## Evaluation

### Performance:
[ x ] - learning curves and r-squared curves (train,val)
[ x ] - test set: eval models, store outputs (hidden layer and readout), perhaps for ~1- samples per condition.
[ x ] - plot ground truth xy coordinates with small scatter plots of predicted coordinates

### Inner workings:
[ X ] - PCA at each timepoint, then plot timeseries of principal components, colour coded by start/ goal location and direction
[ X ] - linear decoding of task features
[ ] - "RSA": marginalise irrelevant dimensions, one RDM per start loc, goal loc, goal direction
[ ] - RSA: model rdms based on xy target locations and direction (angles) the latter with cosine dissimilarity