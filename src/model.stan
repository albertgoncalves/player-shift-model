data {
    int<lower=1> n_players;
    int<lower=1> n_shifts;

    array[n_shifts] int<lower=1> duration;

    array[n_shifts, 5] int<lower=1, upper=n_players> home_players;
    array[n_shifts, 5] int<lower=1, upper=n_players> away_players;

    array[n_shifts] int<lower=0> home_shots;
    array[n_shifts] int<lower=0> away_shots;
}

parameters {
    real intercept;
    real home_advantage;

    vector[n_players] offense;
    vector[n_players] defense;
}

model {
    intercept ~ normal(0.0, 1.0);
    home_advantage ~ normal(0.0, 1.0);

    offense ~ normal(0.0, 1.0);
    defense ~ normal(0.0, 1.0);

    vector[n_shifts] home_alpha;
    vector[n_shifts] away_alpha;

    home_alpha = rep_vector(intercept + home_advantage, n_shifts);
    away_alpha = rep_vector(intercept, n_shifts);

    for (i in 1:n_shifts) {
        for (j in 1:5) {
            home_alpha[i] += offense[home_players[i, j]] + defense[away_players[i, j]];
            away_alpha[i] += offense[away_players[i, j]] + defense[home_players[i, j]];
        }
    }

    home_shots ~ binomial_logit(duration, home_alpha);
    away_shots ~ binomial_logit(duration, away_alpha);
}

generated quantities {
    array[n_shifts] real<lower=0> home_shots_check;
    array[n_shifts] real<lower=0> away_shots_check;

    {
        vector[n_shifts] home_alpha;
        vector[n_shifts] away_alpha;

        home_alpha = rep_vector(intercept + home_advantage, n_shifts);
        away_alpha = rep_vector(intercept, n_shifts);

        for (i in 1:n_shifts) {
            for (j in 1:5) {
                home_alpha[i] += offense[home_players[i, j]] + defense[away_players[i, j]];
                away_alpha[i] += offense[away_players[i, j]] + defense[home_players[i, j]];
            }
        }

        home_shots_check = binomial_rng(duration, inv_logit(home_alpha));
        away_shots_check = binomial_rng(duration, inv_logit(away_alpha));
    }
}
