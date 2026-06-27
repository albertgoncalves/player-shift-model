data {
    int<lower=1> n_players;
    int<lower=1> n_shifts;

    array[n_shifts] int<lower=1> time;

    array[n_shifts] int<lower=1, upper=n_players> home_player_1;
    array[n_shifts] int<lower=1, upper=n_players> home_player_2;
    array[n_shifts] int<lower=1, upper=n_players> home_player_3;
    array[n_shifts] int<lower=1, upper=n_players> home_player_4;
    array[n_shifts] int<lower=1, upper=n_players> home_player_5;

    array[n_shifts] int<lower=1, upper=n_players> away_player_1;
    array[n_shifts] int<lower=1, upper=n_players> away_player_2;
    array[n_shifts] int<lower=1, upper=n_players> away_player_3;
    array[n_shifts] int<lower=1, upper=n_players> away_player_4;
    array[n_shifts] int<lower=1, upper=n_players> away_player_5;

    array[n_shifts] int<lower=0> home_shots;
    array[n_shifts] int<lower=0> away_shots;
}

parameters {
    real home_advantage;

    real<lower=0> sigma_offense;
    real<lower=0> sigma_defense;

    vector[n_players] offense;
    vector[n_players] defense;
}

model {
    sigma_offense ~ exponential(1.0);
    sigma_defense ~ exponential(1.0);

    offense ~ normal(0.0, sigma_offense);
    defense ~ normal(0.0, sigma_defense);

    home_advantage ~ normal(0.0, 1.0);

    for (i in 1:n_shifts) {
        home_shots[i] ~ binomial_logit(
            time[i],

            offense[home_player_1[i]]
            + offense[home_player_2[i]]
            + offense[home_player_3[i]]
            + offense[home_player_4[i]]
            + offense[home_player_5[i]]

            + defense[away_player_1[i]]
            + defense[away_player_2[i]]
            + defense[away_player_3[i]]
            + defense[away_player_4[i]]
            + defense[away_player_5[i]]

            + home_advantage
        );
        away_shots[i] ~ binomial_logit(
            time[i],

            offense[away_player_1[i]]
            + offense[away_player_2[i]]
            + offense[away_player_3[i]]
            + offense[away_player_4[i]]
            + offense[away_player_5[i]]

            + defense[home_player_1[i]]
            + defense[home_player_2[i]]
            + defense[home_player_3[i]]
            + defense[home_player_4[i]]
            + defense[home_player_5[i]]
        );
    }
}
