currentData = read.csv('pyramidDataset.csv', header = T)

head(currentData,30)
practice = glm(WS ~ .-OPS, data = df.model, family = 'binomial')

summary(practice)

B = 10000
summary(currentData)



game.prob = function(play){
  teams = c("TBR","OAK")
  series = sample(teams, play, replace = T, prob = c(.300, .70))
  oak.win = length(series[series == 'OAK'])
  oak.win
}


series.outcome = replicate(B, game.prob(7))
prob.oak.win = length(series.outcome[series.outcome > 2]) / length(series.outcome)

prob.oak.win
