CREATE TABLE games (
    game_id TEXT PRIMARY KEY,
    season INT NOT NULL,
    week INT NOT NULL,
    home_team TEXT NOT NULL,
    away_team TEXT NOT NULL,
    spread_line REAL
);

CREATE TABLE plays (
    game_id TEXT NOT NULL REFERENCES games(game_id),
    play_id INT NOT NULL,
    qtr INT,
    game_seconds_remaining INT,
    postream TEXT,
    home_score INT,
    away_score INT,
    down INT,
    ydstogo INT,
    yardline_100 INT,
    description TEXT,
    PRIMARY KEY (game_id, play_id)
);

CREATE TABLE win_probability (
    game_id TEXT NOT NULL,
    play_id INT NOT NULL,
    model_version TEXT NOT NULL,
    home_wp REAL NOT NULL,
    away_wp REAL NOT NULL,
    scored_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (game_id, play_id, model_version),
    FOREIGN KEY (game_id, play_id) REFERENCES plays(game_id, play_id)
);