library(dplyr)
library(readr)
library(ggplot2)
library(jsonlite)


## loader data
df <- read_delim("../data/user_games.csv", delim="\t") %>%
  rename(user_id = user_steamid, 
         game_id = appid)

## how many in total?
df %>%
  summarise(total_users = n_distinct(user_id),
            total_games = n_distinct(game_id))

## how many games in average?
df %>%
  group_by(user_id) %>%
  summarise(n = n()) %>%
  arrange(-n) %>%
  View()
  summarise(avg_games = mean(n),
            min_games = min(n),
            max_games = max(n))


## how many games per user?
p <- df %>%
  group_by(user_id) %>%
  summarise(game_cnt = n()) %>%
  ggplot() +
  geom_bar(aes(x=reorder(user_id, -game_cnt), y=game_cnt), 
           stat="identity",
           fill="#C80000",
           color="#C80000") +
  scale_y_continuous(breaks=seq(0, 600, by= 100)) +
  xlab("Users") +
  ylab("Number of Games") +
  ggtitle("Number of Games per User") +
  theme_minimal() +
  theme(axis.text.x = element_blank(),
        axis.ticks.x = element_blank(),
        panel.background = element_blank(),
        panel.grid.major = element_blank(),
        panel.grid.minor = element_blank())

ggsave(file="games_per_user.png", plot=p, width=4, height=4)

## how many users per game?
game_users <- df %>%
  group_by(game_id) %>%
  summarise(user_cnt = n())

p <- game_users %>%
  ggplot() +
  geom_bar(aes(x=reorder(game_id, -user_cnt), y=user_cnt), 
           stat="identity",
           fill="#C80000",
           color="#C80000") +
  scale_y_continuous(breaks=seq(0, 250, by= 50)) +
  xlab("Games") +
  ylab("Number of Users") +
  ggtitle("Number of Users per Game") +
  theme_minimal() +
  theme(axis.text.x = element_blank(),
        axis.ticks.x = element_blank(),
        panel.background = element_blank(),
        panel.grid.major = element_blank(),
        panel.grid.minor = element_blank())

ggsave(file="users_per_game.png", plot=p, width=4, height=4)

## load game data (only used for game names here)
games_r <- fromJSON("../data/full_games.json")
games_tbl <- as_tibble(games_r)
## ony names, without missing ones
game_names <- games_tbl %>%
  mutate(game_id = as.numeric(id)) %>%
  select(game_id, name) %>%
  filter(nchar(name) > 0)

## game co-occurences
total_users = count(unique(df["user_id"])) %>% first()
coocs_mat <- df %>%
  left_join(df, by=c("user_id")) %>%
  filter(!is.na(game_id.y)) %>%
  filter(game_id.x != game_id.y) %>%
  filter(game_id.x != 391220 & game_id.y != 391220) %>%
  group_by(game_id.x, game_id.y) %>%
  summarise(coocs = n()) %>%
  left_join(game_users, by=c("game_id.x" = "game_id")) %>%
  group_by(game_id.x, game_id.y) %>%
  mutate(cooc_factor = coocs / user_cnt) %>%
  ungroup()

## see top co-occuring games
coocs_mat %>%
  filter(coocs > 15) %>%
  arrange(-cooc_factor) %>%
  head(200) %>%
  inner_join(game_names, by=c("game_id.x" = "game_id")) %>%
  inner_join(game_names, by=c("game_id.y" = "game_id")) %>%
  select(game1 = name.x, game2 = name.y) %>%
  View()

## see least co-occuring games
coocs_mat %>%
  filter(coocs > 15) %>%
  arrange(cooc_factor) %>%
  head(200) %>%
  inner_join(game_names, by=c("game_id.x" = "game_id")) %>%
  inner_join(game_names, by=c("game_id.y" = "game_id")) %>%
  select(game1 = name.x, game2 = name.y) %>%
  View()

library(datapasta)
## copy data e.g. from Excel sheet and just paste it with CTRL + shift + V:
df <- tibble::tribble(
  ~syns_cnt, ~docs_cnt, ~quality_negatives, ~remove_positive_ids,
         1L,        1L,                  0,                    0,
         NA,        NA,                 NA,                   NA,
         NA,        NA,                 NA,                   NA,
         1L,        1L,                0.9,                  0.9,
         1L,        2L,                0.9,                  0.9,
         2L,        1L,                0.9,                  0.9,
         2L,        2L,                0.9,                  0.9,
         2L,        3L,                0.9,                  0.9,
         1L,        1L,                0.9,                  0.9,
         1L,        1L,                0.9,                  0.9,
         1L,        1L,                0.9,                  0.9,
         1L,        1L,                0.9,                  0.9
  )
