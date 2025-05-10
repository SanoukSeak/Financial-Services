
  
# Install and load required packages

  packages <- c("shiny", "dplyr", "ggplot2", "tidytext", "tidyr", "stringr", "lubridate", "jsonlite", "countrycode")
lapply(packages, function(pkg) {
  if (!requireNamespace(pkg, quietly = TRUE)) install.packages(pkg)
  library(pkg, character.only = TRUE)
})

# Read JSON file

airbnb_file <- file("airbnb_listings.json", "r")
airbnb_collection <- stream_in(airbnb_file, flatten = TRUE)
close(airbnb_file)

# Separate location into columns

airbnb_loc <- airbnb_collection %>%
  separate(`host.host_location`, into = c("city", "state/province", "country"), sep = ",", fill = "right", extra = "merge")

# Clean and format country column

airbnb_loc$country <- trimws(airbnb_loc$country)
airbnb_loc$country <- str_to_title(airbnb_loc$country)
airbnb_loc$country_code <- countrycode(airbnb_loc$country, origin = "country.name", destination = "iso3c")

# Clean numeric columns (price, etc.)

airbnb_loc <- airbnb_loc %>%
  mutate(
    price = as.numeric(`price.$numberDecimal`),
    last_review = as.Date(last_review),
    reviews = as.character(reviews)
  )

  
# Create a dashboard
  
  
library(shiny)
library(dplyr)
library(ggplot2)
library(tidytext)
library(tidyr)
library(stringr)
library(lubridate)
library(ggraph)
library(igraph)

# Load AFINN and stop words
afinn <- get_sentiments("afinn")
data("stop_words")

# UI
ui <- fluidPage(
  titlePanel("Airbnb Client-Property Analysis (AFINN only, no stop words)"),
  
  fluidRow(
    column(12, align = "center",
           sliderInput("year_slider", "Select Year",
                       min = 2012, max = 2019, value = 2015, step = 1,
                       width = "80%")
    )
  ),
  
  fluidRow(
    column(12, align = "center",
           actionButton("us_button", "United States", class = "btn btn-primary", style = "margin: 10px;"),
           actionButton("es_button", "Spain", class = "btn btn-primary", style = "margin: 10px;"),
           actionButton("aus_button", "Australia", class = "btn btn-primary", style = "margin: 10px;")
    )
  ),
  
  hr(),
  
  fluidRow(
    column(6, plotOutput("correlation_network")),
    column(6, plotOutput("D3V2"))
  ),
  hr(),
  fluidRow(
    column(6, plotOutput("D3V3")),
    column(6, plotOutput("D3V4"))
  )
)

# Server
server <- function(input, output, session) {
  selected_country <- reactiveVal("United States")
  
  observeEvent(input$us_button, { selected_country("United States") })
  observeEvent(input$es_button, { selected_country("Spain") })
  observeEvent(input$aus_button, { selected_country("Australia") })
  
  filtered_data <- reactive({
    req(airbnb_loc)
    airbnb_loc %>%
      filter(country == selected_country(),
             year(as.Date(`last_review`)) == input$year_slider) %>%
      mutate(
        `price.$numberDecimal` = as.numeric(`price.$numberDecimal`),
        `security_deposit.$numberDecimal` = as.numeric(`security_deposit.$numberDecimal`),
        accommodates = as.numeric(accommodates),
        reviews = as.character(reviews)
      ) %>%
      filter(!is.na(`price.$numberDecimal`), `price.$numberDecimal` < 1500)
  })
  
  # Correlation network
  output$correlation_network <- renderPlot({
    req(nrow(filtered_data()) > 0)
    
    top_des_tokens <- filtered_data() %>%
      unnest_tokens(word, description) %>%
      anti_join(stop_words, by = "word") %>%
      count(word, sort = TRUE) %>%
      mutate(rank = row_number(), tf = n / sum(n), zipf_score = tf / rank) %>%
      slice_max(zipf_score, n = 20) %>%
      mutate(word_desc = paste0("desc_", word))
    
    rev_tokens <- filtered_data() %>%
      unnest_tokens(word, reviews) %>%
      anti_join(stop_words, by = "word") %>%
      inner_join(afinn, by = "word") %>%
      group_by(word) %>%
      summarise(sentiment = mean(value), n = n(), .groups = "drop") %>%
      filter(n > 30) %>%
      mutate(word_rev = paste0("rev_", word))
    
    edges <- rev_tokens %>%
      inner_join(top_des_tokens, by = "word") %>%
      filter(abs(sentiment) > 0.5) %>%
      select(from = word_desc, to = word_rev, sentiment) %>%
      mutate(color = ifelse(sentiment > 0, "#00b146", "red"))
    
    nodes <- bind_rows(
      top_des_tokens %>%
        transmute(name = word_desc, label = word, type = "description", color = "blue", shape = "triangle"),
      rev_tokens %>%
        transmute(name = word_rev,
                  label = word,
                  type = "review",
                  color = case_when(
                    sentiment > 0 ~ "#00b146",
                    sentiment < 0 ~ "red",
                    TRUE ~ "grey"
                  ),
                  shape = "circle")
    )
    
    graph <- graph_from_data_frame(edges, vertices = nodes, directed = FALSE)
    
    ggraph(graph, layout = "fr") +
      geom_edge_link(aes(color = color), show.legend = TRUE) +
      geom_node_point(aes(color = color, shape = shape), size = 3) +
      geom_node_text(aes(label = label), repel = TRUE, size = 3) +
      scale_color_manual(
        values = c("blue" = "blue", "#00b146" = "#00b146", "red" = "red"),
        name = "Sentiment",
        labels = c("blue" = "Description Tokens",
                   "#00b146" = "Positive Reviews",
                   "red" = "Negative Reviews")
      ) +
      scale_shape_manual(values = c("triangle" = 17, "circle" = 16)) +
      guides(shape = "none") +
      labs(title = "Correlation Network of Description and Review Tokens") +
      theme_void() +
      theme(plot.title = element_text(face = "bold", size = 13))
  })
  
  # Sentiment Scatter
  output$D3V2 <- renderPlot({
    data <- filtered_data()
    
    review_tokens <- data %>%
      unnest_tokens(word, reviews) %>%
      anti_join(stop_words, by = "word") %>%
      inner_join(afinn, by = "word")
    
    word_price_data <- review_tokens %>%
      group_by(`price.$numberDecimal`, word, value) %>%
      summarise(frequency = n(), .groups = "drop") %>%
      mutate(sentiment_color = ifelse(value >= 0, "green", "red"))
    
    ggplot(word_price_data, aes(x = `price.$numberDecimal`, y = frequency, color = sentiment_color)) +
      geom_point(alpha = 0.6) +
      scale_color_manual(values = c("green" = "#00b146", "red" = "red"),
                         labels = c("green" = "Positive", "red" = "Negative")) +
      labs(
        title = "AFINN Token Frequency vs. Price",
        x = "Price (USD)", y = "Frequency of Review Tokens", color = "Sentiment"
      ) +
      theme_minimal() +
      theme(
        plot.title = element_text(face = "bold", size = 13),
        axis.text = element_text(size = 12, color = "black"),
        axis.title = element_text(size = 14, color = "black"),
        legend.title = element_text(size = 12),
        legend.text = element_text(size = 10)
      )
  })
  
  # Radial TF-IDF AFINN
  output$D3V3 <- renderPlot({
    data <- filtered_data()
    
    review_tokens <- data %>%
      unnest_tokens(word, reviews) %>%
      anti_join(stop_words, by = "word") %>%
      inner_join(afinn, by = "word")
    
    review_tf_idf <- review_tokens %>%
      count(listing_url, word, sort = TRUE) %>%
      bind_tf_idf(word, listing_url, n) %>%
      group_by(word) %>%
      summarise(mean_tf_idf = mean(tf_idf), .groups = "drop") %>%
      arrange(desc(mean_tf_idf)) %>%
      inner_join(afinn, by = "word") %>%
      slice_max(mean_tf_idf, n = 8)
    
    review_tf_idf$word <- factor(review_tf_idf$word, levels = review_tf_idf$word)
    
    ggplot(review_tf_idf, aes(x = word, y = value, fill = value)) +
      geom_bar(stat = "identity", width = 1, color = "black") +
      coord_polar(start = 0) +
      scale_y_continuous(limits = c(-5, 5), breaks = seq(-5, 5, 1)) +
      scale_fill_gradient2(low = "red", mid = "yellow", high = "green", midpoint = 0) +
      labs(
        title = "Top 8 Review Tokens by TF-IDF and AFINN Sentiment",
        x = NULL, y = "AFINN Sentiment Score"
      ) +
      theme_minimal() +
      theme(
        axis.text.x = element_text(size = 12, color = "black", face = "bold"),
        axis.title.y = element_text(size = 14, color = "black"),
        plot.title = element_text(face = "bold", size = 14, hjust = 0.5),
        panel.grid.major = element_line(color = "gray80", size = 0.5),
        panel.grid.minor = element_line(color = "gray90", size = 0.3)
      )
  })
  
  # Bubble chart
  output$D3V4 <- renderPlot({
    data <- filtered_data() %>%
      filter(!is.na(`security_deposit.$numberDecimal`), `security_deposit.$numberDecimal` < 5000, !is.na(accommodates))
    
    fill_colors <- c(
      "Entire home/apt" = "#1E90FF",
      "Private room" = "#FEBA4F",
      "Shared room" = "#16E3B0",
      "Hotel room" = "#8A2BE2"
    )
    border_colors <- c(
      "Entire home/apt" = "#000080",
      "Private room" = "#F76806",
      "Shared room" = "#008B8B",
      "Hotel room" = "#9370DB"
    )
    
    ggplot(data, aes(x = `price.$numberDecimal`, y = `security_deposit.$numberDecimal`)) +
      geom_point(aes(color = room_type, fill = room_type, size = accommodates),
                 shape = 21, stroke = 0.73) +
      scale_fill_manual(values = fill_colors, name = "Room Type") +
      scale_color_manual(values = border_colors, name = "Room Type") +
      scale_size_continuous(name = "Accommodates", range = c(1, 6),
                            breaks = c(1, 2, 4, 6, 8),
                            labels = c("1", "2", "4", "6", "8+")) +
      labs(
        title = "Security Deposit vs. Price by Room Type and Accommodates",
        x = "Price (USD)", y = "Security Deposit (USD)"
      ) +
      theme_minimal() +
      theme(
        plot.title = element_text(face = "bold", size = 14, margin = margin(b = 20)),
        axis.title.x = element_text(size = 14, color = "black", margin = margin(t = 15)),
        axis.title.y = element_text(size = 14, color = "black", margin = margin(r = 15)),
        axis.text.x = element_text(size = 12, color = "black", margin = margin(t = 10)),
        axis.text.y = element_text(size = 12, color = "black", margin = margin(r = 10)),
        legend.title = element_text(size = 12),
        legend.text = element_text(size = 10)
      )
  })
}

# Run the app
#shinyApp(ui = ui, server = server)
