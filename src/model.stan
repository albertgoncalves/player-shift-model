data {
    int<lower=1> n_players;
    int<lower=1> n_shifts;

    array[n_shifts] int<lower=1> durations;

    array[n_shifts, 5] int<lower=1, upper=n_players> home_players;
    array[n_shifts, 5] int<lower=1, upper=n_players> away_players;

    array[n_shifts] int<lower=0> home_shots;
    array[n_shifts] int<lower=0> away_shots;
}

parameters {
    real intercept;
    real home_advantage;

    real<lower=0> sigma_intercept;
    real<lower=0> sigma_home_advantage;
    real<lower=0> sigma_offense;
    real<lower=0> sigma_defense;

    vector[n_players] offense;
    vector[n_players] defense;
}

model {
    sigma_intercept ~ exponential(0.1);
    sigma_home_advantage ~ exponential(0.1);
    sigma_offense ~ exponential(0.1);
    sigma_defense ~ exponential(0.1);

    intercept ~ normal(0.0, sigma_intercept);
    home_advantage ~ normal(0.0, sigma_home_advantage);

    offense ~ normal(0.0, sigma_offense);
    defense ~ normal(0.0, sigma_defense);

    real home_alpha;
    real away_alpha;

    for (i in 1:n_shifts) {
        home_alpha = 0;
        away_alpha = 0;

        for (j in 1:5) {
            home_alpha += offense[home_players[i, j]] + defense[away_players[i, j]];
            away_alpha += offense[away_players[i, j]] + defense[home_players[i, j]];
        }

        home_shots[i] ~ binomial_logit(durations[i], intercept + home_alpha + home_advantage);
        away_shots[i] ~ binomial_logit(durations[i], intercept + away_alpha);
    }
}
