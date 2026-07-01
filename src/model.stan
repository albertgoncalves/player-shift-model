data {
    int<lower=1> n_players;
    int<lower=1> n_shifts;
    int<lower=1> n_train;

    array[n_shifts] int<lower=1> duration;

    array[n_shifts, 5] int<lower=1, upper=n_players> home_players;
    array[n_shifts, 5] int<lower=1, upper=n_players> away_players;

    array[n_shifts] int<lower=0> home_shots;
    array[n_shifts] int<lower=0> away_shots;
}

transformed data {
    int<lower=1> n_test;
    n_test = n_shifts - n_train;

    array[n_train] int<lower=1> duration_train;
    array[n_test]  int<lower=1> duration_test;

    array[n_train] int<lower=0> home_shots_model;
    array[n_train] int<lower=0> away_shots_model;

    for (i in 1:n_train) {
        duration_train[i] = duration[i];

        home_shots_model[i] = home_shots[i];
        away_shots_model[i] = away_shots[i];
    }
    for (i in 1:n_test) {
        duration_test[i] = duration[i + n_train];
    }
}

parameters {
    real intercept;
    real home_advantage;

    vector[n_players] offense;
    vector[n_players] defense;
}

model {
    intercept      ~ normal(0.0, 1.0);
    home_advantage ~ normal(0.0, 1.0);

    offense ~ normal(0.0, 1.0);
    defense ~ normal(0.0, 1.0);

    vector[n_train] home_alpha;
    vector[n_train] away_alpha;

    home_alpha = rep_vector(intercept + home_advantage, n_train);
    away_alpha = rep_vector(intercept, n_train);

    for (i in 1:n_train) {
        for (j in 1:5) {
            home_alpha[i] += offense[home_players[i, j]] + defense[away_players[i, j]];
            away_alpha[i] += offense[away_players[i, j]] + defense[home_players[i, j]];
        }
    }

    home_shots_model ~ binomial_logit(duration_train, home_alpha);
    away_shots_model ~ binomial_logit(duration_train, away_alpha);
}

generated quantities {
    array[n_train] real<lower=0> home_shots_train;
    array[n_train] real<lower=0> away_shots_train;

    array[n_test] real<lower=0> home_shots_test;
    array[n_test] real<lower=0> away_shots_test;

    {
        vector[n_train] home_alpha;
        vector[n_train] away_alpha;

        home_alpha = rep_vector(intercept + home_advantage, n_train);
        away_alpha = rep_vector(intercept, n_train);

        for (i in 1:n_train) {
            for (j in 1:5) {
                home_alpha[i] += offense[home_players[i, j]] + defense[away_players[i, j]];
                away_alpha[i] += offense[away_players[i, j]] + defense[home_players[i, j]];
            }
        }

        home_shots_train = binomial_rng(duration_train, inv_logit(home_alpha));
        away_shots_train = binomial_rng(duration_train, inv_logit(away_alpha));
    }

    {
        vector[n_test] home_alpha;
        vector[n_test] away_alpha;

        home_alpha = rep_vector(intercept + home_advantage, n_test);
        away_alpha = rep_vector(intercept, n_test);

        for (i in 1:n_test) {
            for (j in 1:5) {
                home_alpha[i] += offense[home_players[i + n_train, j]]
                              +  defense[away_players[i + n_train, j]];
                away_alpha[i] += offense[away_players[i + n_train, j]]
                              +  defense[home_players[i + n_train, j]];
            }
        }

        home_shots_test = binomial_rng(duration_test, inv_logit(home_alpha));
        away_shots_test = binomial_rng(duration_test, inv_logit(away_alpha));
    }
}
