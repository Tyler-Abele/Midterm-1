# DS 4320 Project 1: Conflict Prediction in the Middle East

This repository contains a fully established secondary dataset built using the relational model to predict the likelihood of elevated armed conflict across countries in the Middle East. To make the data more consistent and modeling-friendly, the project standardizes multiple sources with different time granularities into a common **country-year** analytical unit. Event-level conflict data from ACLED and GDELT are preserved in raw tables, then aggregated into annual country-level features and combined with yearly structural indicators such as military spending and Fragile States Index measures. A machine learning pipeline loads the data into DuckDB, prepares the final modeling table, and trains classification models including Random Forest and XGBoost to predict whether a country will experience elevated conflict in the following year. The goal is to support interpretable early warning analysis for policymakers, defense analysts, and humanitarian organizations.

## Author
- Name: Tyler Abele
- NetID: xxe9ff

**DOI:** [(https://zenodo.org/badge/1189040869.svg)](https://doi.org/10.5281/zenodo.19363064)

## Quick Links:
| **Resource** | **Link** |
| --- | --- |
| Press Release | [link](./press-release.md) |
| Data (OneDrive) | [link](https://myuva-my.sharepoint.com/:f:/g/personal/xxe9ff_virginia_edu/IgDJSQnz8iQIRrmr4j2IUB8sAeYSPlwAUdMYLu0Idl7zoms?e=DfhR5o)|
| Data (repo) | [link](./data/) |
| Pipeline Notebook | [link](./pipeline/master_pipeline.ipynb)|
| Pipeline Markdown | [link](./pipeline/master_pipeline.md)|
| Liscense | [link](./LICENSE)|

### License

This project is Liscened under the MIT Liscense. [link](./LICENSE)

## Problem Definition

### General Problem
Predicting armed conflict

### Specific Problem
Can we predict whether a Middle Eastern country will experience elevated conflict in the following year using structural indicators, media-event signals, and recent conflict activity?

### Rationale for Refinement
The project was refined to focus on **country-year conflict prediction in the Middle East**. I originally considered a more granular subnational design, but the available datasets operate at different temporal levels, including daily event-level data and yearly structural indicators. To make the relational model cleaner and the machine learning pipeline more interpretable, I standardized the analysis to a single country-year grain. This allows event-level ACLED and GDELT data to be aggregated into annual country features and joined cleanly with annual military spending and Fragile States Index data. The Middle East remains a strong setting for this problem because it includes repeated cycles of interstate conflict, civil conflict, protests, and state fragility, making it a meaningful domain for conflict forecasting.

## Motivation
This project is personally meaningful to me because my father served in Iraq during the War on Terror when I was a child. Since then, I have been interested in the region and the conditions that contribute to instability and violence. Armed conflict causes enormous human suffering, displacement, and economic destruction. If publicly available data can be organized into a useful early warning system, even at the annual country level, it could help analysts and decision-makers better understand where conflict risk is rising. This project explores whether conflict event data, media-event data, and structural national indicators can be combined into a relational dataset that supports predictive modeling of future conflict.

## Press Release 
**Headline** New Conflict Hotspot Tool Could Save Lives in the Middle East.

see the full press release: [press-release.md](./press-release.md)

## Domain Expositon

### Terminology

### Terminology

| **Term** | **Definition** |
|---|---|
| ACLED | Armed Conflict Location and Event Data Project — an organization that collects and publishes event-level data on political violence and protest activity worldwide. |
| GDELT | Global Database of Events, Language, and Tone — a large-scale event database that captures coded geopolitical events, media attention, and sentiment-related measures from global news sources. |
| Fragile States Index (FSI) | An annual index published by the Fund for Peace that measures state vulnerability using social, economic, and political indicators. |
| Military Expenditure | Annual spending by a country on its armed forces, typically reported in current U.S. dollars. |
| Country-Year | The analytical unit used in this project, where each row represents one country in one year. |
| Event-Level Data | Data recorded at the level of individual events, such as a single protest, battle, or coded media event. |
| Aggregation | The process of summarizing lower-level data, such as event-level records, into higher-level features such as annual country totals. |
| Fact Table | A table containing measured events or observations, such as ACLED events, GDELT events, or country-year features. |
| Dimension Table | A table containing descriptive reference information, such as country names, region, or income group classifications. |
| Political Violence | Violent conflict-related activity, including battles, explosions or remote violence, and violence against civilians. |
| Battles | ACLED events involving direct violent interaction between organized armed groups. |
| Explosions/Remote Violence | ACLED events involving bombs, shelling, missiles, airstrikes, drones, or other forms of violence delivered from a distance. |
| Violence Against Civilians | ACLED events in which armed actors intentionally target unarmed civilians. |
| Protests | ACLED events involving nonviolent public demonstrations. |
| Riots | ACLED events involving violent disorder by groups such as mobs or demonstrators. |
| Fatalities | The number of reported deaths associated with a conflict event. |
| Event Count | The total number of recorded events for a country in a given year. |
| Conflict Predictor | A model designed to estimate the likelihood that a country will experience elevated conflict in a future period. |
| Target Variable | The outcome the model is trying to predict; in this project, whether elevated conflict occurs in the following year. |
| Binary Classification | A machine learning task with two possible outcomes, here coded as conflict next year or no elevated conflict next year. |
| Feature | An input variable used by a model to make predictions. |
| Feature Engineering | The process of creating useful predictors from raw data, such as annual event totals, annual fatalities, or average media tone. |
| Temporal Granularity | The time scale at which data are recorded, such as daily event-level data or yearly structural indicators. |
| Temporal Leakage | A modeling problem where information from the future is allowed to influence the training process, leading to overly optimistic results. |
| Chronological Train/Test Split | A modeling approach where earlier years are used for training and later years are used for testing in order to preserve time order. |
| Random Forest | An ensemble machine learning method based on many decision trees, used here for binary classification. |
| XGBoost | A gradient-boosted tree model that is often effective for structured tabular prediction tasks. |
| Class Imbalance | A situation where one outcome class occurs much more frequently than the other, such as many more non-conflict cases than conflict cases. |
| Imputation | The process of filling in missing data values, such as replacing missing numeric values with the median. |
| Feature Importance | A measure of how much each predictor contributes to a model’s decisions. |
| Confusion Matrix | A table showing correct and incorrect predictions broken into true positives, true negatives, false positives, and false negatives. |
| Precision | The share of predicted positive cases that were actually positive. |
| Recall | The share of actual positive cases that the model correctly identified. |
| F1 Score | A metric that balances precision and recall into a single value. |
| ROC-AUC | A metric that summarizes how well a classifier separates positive and negative cases across probability thresholds. |
| DuckDB | An in-process analytical SQL database used in this project for loading, transforming, joining, and querying the relational dataset. |
| Codebook | Documentation that explains variables, event types, coding choices, and other metadata in a dataset. |
| Parquet | A columnar file format commonly used for efficient storage and analysis of large datasets. |

## Domain Overview
The domain of this project is conflict analysis and early warning within the broader field of defense and security studies. This field sits at the intersection of political science, international relations, and data science, and is concerned with understanding the patterns, drivers, and dynamics of political violence and armed conflict around the world. In practice, this involves the systematic collection and analysis of event-level conflict data, who attacked whom, where, when, and with what consequences, alongside contextual indicators like military spending, economic conditions, and state fragility, to identify regions at risk of escalating violence. The insights generated from this kind of analysis are used by defense analysts, intelligence agencies, humanitarian organizations, and policymakers to anticipate crises, allocate resources, and inform intervention decisions.

## Background Reading

All Background reading materials are stored in the [`readings`](./readings/) folder and the UVA OneDrive folder here: [`UVA OneDrive Folder`](https://myuva-my.sharepoint.com/:f:/r/personal/xxe9ff_virginia_edu/Documents/Midterm%201?csf=1&web=1&e=sJZkTj).

| **Title** | **Description** | **repo link** | **onedrive link**
| --- | --- | --- | --- |
| ACLED CAST Methodology | background on conflict forecasting methods using event data. | [repo](./readings/ACLED_CAST_Methodology.pdf) |[onedrive](https://myuva-my.sharepoint.com/:b:/r/personal/xxe9ff_virginia_edu/Documents/Midterm%201/readings/ACLED-CAST-Methodology-%E2%80%93-July-2023.pdf?csf=1&web=1&e=XyV9yL) |
| FSI Methodology | explanation of how fragility indicators are constructed and interpreted. | [repo](./readings/FSI_Methodology.pdf) | [onedrive](https://myuva-my.sharepoint.com/:b:/r/personal/xxe9ff_virginia_edu/Documents/Midterm%201/readings/FSI-Methodology.pdf?csf=1&web=1&e=6eVYdt) |
| SIPRI Military Expendature Database |regional background on militarization and security dynamics. | [repo](./readings/SIPRI_world_expendature.pdf) | [onedrive](https://myuva-my.sharepoint.com/:b:/r/personal/xxe9ff_virginia_edu/Documents/Midterm%201/readings/SIPRI_world_expendature.pdf?csf=1&web=1&e=u0Zfvv) |
| SIPRI Middle East Military Spending and Arms Transfers | current background on defense spending, including the Middle East. | [repo](./readings/SIPRI.pdf) | [onedrive](https://myuva-my.sharepoint.com/:b:/r/personal/xxe9ff_virginia_edu/Documents/Midterm%201/readings/SIPRI.pdf?csf=1&web=1&e=Arv6AS) |
| World Bank Economic Note on the Middle East Conflict | regional context on the socioeconomic consequences of conflict. | [repo](./readings/WorldBank_Palestine.pdf) | [onedrive](https://myuva-my.sharepoint.com/:b:/r/personal/xxe9ff_virginia_edu/Documents/Midterm%201/readings/WorldBank_Palestine.pdf?csf=1&web=1&e=gkg7vk) |

## Data Creation

### Data Aquisiton Process

Conflict prediction sits at the intersection of political science, data science, and humanitarian operations. The field has evolved from qualitative expert assessments to increasingly quantitative approaches that leverage georeferenced event data, satellite imagery, economic indicators, and social media signals. Organizations like the UN, World Bank, and International Crisis Group rely on early warning systems to allocate resources and guide intervention. The Middle East, spanning countries from Egypt to Iran, presents unique modeling challenges due to overlapping civil wars, proxy conflicts, sectarian tensions, and rapid political transitions. Key data sources include ACLED, UCDP, the World Bank Development Indicators, and the Fragile States Index (created by the Fund for Peace).

### Code

| File | Description | Link |
| ---- | ----------- | --- |
| raw_data_loading | loads all of the raw data into the duckdb database | [Link](./code/raw_data_loading.ipynb) |
| dim_country.ipynb | creates the dim country database, has general country data | [Link](./code/dim_country.ipynb) |
| fact_acled_event.ipynb | holds all of the import acled data | [Link](./code/fact_acled_event.ipynb) |
| fact_gdelt_event.ipynb | holds all of the imported gdelt data | [Link](./code/fact_gdelt_event.ipynb) |
| fact_country_year.ipynb | aggregates many of the other raw data tables into a single table that is easy to use for anayltics | [Link](./code/fact_country_year.ipynb) |
| master_pipeline | **Does everything including raw data loading** | [Link](./pipeline/master_pipeline.ipynb) |


**Note:** If you are not using the master pipeline, you need to do raw data loading first. If you are using master pipeline then you do not need to do anything else. Note for all of these you must download the raw data files and place them under the data tab. I would do this for you but the file sizes are too big for github.

### Bias Identification

Several important sources of bias and uncertainty are present in this dataset. First, conflict event data such as ACLED may underreport events or fatalities in regions with limited media access, restricted reporting, or active censorship. Second, GDELT reflects media and source coverage rather than direct ground truth, so countries with heavier international media attention may appear more eventful than countries with similar conditions but weaker coverage. Third, annual military spending figures may be inconsistently reported across countries and may not fully reflect corruption, off-budget spending, or true defense capacity. Finally, the Fragile States Index is a composite indicator and therefore reflects methodological assumptions made by its creators. These issues mean that model outputs should be interpreted as informative signals rather than objective truth.

### Bias Mitigation

To reduce the impact of these biases, I standardized all modeling features to a common country-year grain and used multiple data sources rather than relying on a single dataset. This allows the model to learn from both direct conflict activity and broader structural indicators. I also preserve the raw event-level tables separately from the final modeling table so that aggregation decisions remain transparent and reproducible. In analysis, I treat variables such as media-event counts and military spending as imperfect proxies rather than exact measurements. These limitations are documented in the data dictionary and uncertainty notes for numerical features.

### Critical Decision Rationale

A major design decision in this project was to simplify the schema around a single analytical grain: **country-year**. The original design mixed daily, weekly, and yearly data in ways that made joins ambiguous and would have complicated both the relational model and the machine learning pipeline. To make the project more coherent and achievable, I preserved ACLED and GDELT as raw event-level tables, then aggregated them into yearly country-level features that can be joined with annual structural indicators such as military spending and Fragile States Index variables. I also chose to exclude more complicated extensions, such as subnational forecasting and additional socioeconomic indicators, in order to prioritize a cleaner relational design and a complete end-to-end predictive pipeline. This simplification improves interpretability, reduces duplication risk, and better aligns the schema with the project goal of predicting elevated conflict in the following year.

## Metadata

### Schema

**ER diagram at the logical level showing relationships between tables. Created with Lucidcharts.**
![ERD Diagram](./assets/er_diagram.png)


### Data Tables

| Table | Description | File (OneDrive) | 
|-------|-------------|------ |
| dim_countries | Central lookup table with 15 Middle East countries, ISO/FIPS codes, and World Bank income group | [onedrive](https://myuva-my.sharepoint.com/:x:/r/personal/xxe9ff_virginia_edu/Documents/Midterm%201/data/relational/dim_country.csv?d=w1542462b97054f28ac8515cc05c7d848&csf=1&web=1&e=zzGrmi) |
| fact_acled_events | 144,526 weekly aggregated conflict events from ACLED covering battles, protests, explosions, riots, violence against civilians, and strategic developments across the Middle East (2015–2026) | [onedrive](https://myuva-my.sharepoint.com/:x:/r/personal/xxe9ff_virginia_edu/Documents/Midterm%201/data/relational/fact_acled_event.csv?d=w581c0339bf0740d3a6bf295fc34c0373&csf=1&web=1&e=znyOUF) |
| fact_gdelt_events | 2,148,627 media-sourced conflict events from GDELT with Goldstein conflict intensity scores, average article tone, and media mention counts (2023–2025) | [onedrive](https://myuva-my.sharepoint.com/:x:/r/personal/xxe9ff_virginia_edu/Documents/Midterm%201/data/relational/fact_gdelt_event.csv?d=w7a49e72e00944c3881fb40131f0608f4&csf=1&web=1&e=vSkDpY) |
| fact_country_year | Final modeling table aggregating all sources into one row per country per year with target variable for next-year conflict escalation |[onedrive](https://myuva-my.sharepoint.com/:x:/r/personal/xxe9ff_virginia_edu/Documents/Midterm%201/data/relational/fact_country_year.csv?d=we88acb750686452f99a73b03cd0a9942&csf=1&web=1&e=UhKzCh) |

### Data Dictonary

#### dim_countries

| Feature | Data Type | Description | Example |
|---------|-----------|-------------|---------|
| country_code | str | ISO 3-letter country code, primary key | IRQ |
| country_name | str | Full country name | Iraq |
| fips_code | str | FIPS country code used by GDELT | IZ |
| region | str | Geographic region, always Middle East | Middle East |
| income_group | str | World Bank income classification | Upper middle income |
| lending_category | str | World Bank lending category | IBRD |

#### fact_acled_event

| Feature | Data Type | Description | Example |
|---------|-----------|-------------|---------|
| acled_event_id | int | Auto-generated unique row identifier, primary key | 42357 |
| country_code | str | ISO 3-letter code, foreign key to dim_countries | SYR |
| event_date | str | Week start date of the aggregated event record | 2023-01-07 |
| year | str | Year extracted from event date | 2023 |
| event_type | str | ACLED event category | Battles |
| sub_event_type | str | ACLED sub-category of the event | Armed clash |
| fatalities | int | Number of reported deaths in the event period | 14 |
| population_exposure | int | Estimated civilians exposed to violence in the area | 38447 |
| centroid_latitude | float | Latitude of the event admin region centroid | 33.3128 |
| centroid_longitude | float | Longitude of the event admin region centroid | 44.3615 |

#### fact_gdelt_event

| Feature | Data Type | Description | Example |
|---------|-----------|-------------|---------|
| gdelt_event_id | int | GDELT unique event identifier, primary key | 1078326512 |
| country_code | str | ISO 3-letter code, foreign key to dim_countries | ISR |
| event_date | str | Date of the event in YYYYMMDD format | 20231015 |
| year | str | Year extracted from event date | 2023 |
| event_root_code | str | CAMEO root code classifying the event type (14=Protest, 18=Assault, 19=Fight, 20=Mass violence) | 19 |
| quadclass | int | Conflict/cooperation quadrant: 1=Verbal Cooperation, 2=Material Cooperation, 3=Verbal Conflict, 4=Material Conflict | 4 |
| goldsteinscale | float | Conflict intensity score ranging from -10 (extreme conflict) to +10 (extreme cooperation) | -7.2 |
| nummentions | int | Number of times this event was mentioned across all source articles | 29 |
| numscores | int | Number of distinct news sources reporting this event | 4 |
| avgtone | float | Average sentiment tone of source articles, negative values indicate negative coverage | -5.83 |

#### fact_country_year

| Feature | Data Type | Description | Example |
|---------|-----------|-------------|---------|
| country_code_year | str | Composite primary key combining country code and year | SYR_2023 |
| country_code | str | ISO 3-letter code, foreign key to dim_countries | SYR |
| acled_event_count | int | Total number of ACLED conflict events recorded for the country that year | 4821 |
| acled_fatalities_sum | int | Total reported fatalities from all ACLED events that year | 1293 |
| acled_protest_sum | int | Total number of protest events recorded by ACLED that year | 587 |
| gdelt_event_count | int | Total number of GDELT media-sourced conflict events that year | 28450 |
| gdelt_avg_tone | str | Average media tone across all GDELT events that year, negative = negative coverage | -3.2145 |
| gdelt_avg_goldstein | float | Average Goldstein conflict intensity score across all GDELT events that year | -6.18 |
| gdelt_total_mentions | int | Total media mentions across all GDELT events that year | 142800 |
| gdelt_total_scores | float | Total number of distinct sources across all GDELT events that year | 18450.0 |
| military_spending_current_usd | float | Annual military expenditure in millions of current USD from SIPRI | 5997.34 |
| fsi_total_points | float | Fragile States Index composite score ranging from 0 (most stable) to 120 (most fragile) | 96.2 |
| demographic_pressures | float | FSI S1 indicator measuring population growth, disease, resource scarcity (0-10) | 8.1 |
| refugees_idps | float | FSI S2 indicator measuring displacement and refugee flows (0-10) | 9.6 |
| group_grievances | float | FSI C3 indicator measuring ethnic, religious, or communal tensions (0-10) | 8.7 |
| human_flight | float | FSI E3 indicator measuring emigration and brain drain (0-10) | 6.4 |
| economic_inequality | float | FSI E2 indicator measuring wealth gaps and economic exclusion (0-10) | 7.9 |
| economy | float | FSI E1 indicator measuring economic decline, poverty, and unemployment (0-10) | 9.5 |
| state_legitimacy | float | FSI P1 indicator measuring public trust in government and institutions (0-10) | 9.8 |
| public_services | float | FSI P2 indicator measuring provision of health, education, and infrastructure (0-10) | 9.6 |
| human_rights | float | FSI P3 indicator measuring civil liberties, press freedom, and political rights (0-10) | 9.0 |
| security_apparatus | float | FSI C1 indicator measuring internal security threats and state monopoly on force (0-10) | 9.5 |
| factionalized_elites | float | FSI C2 indicator measuring political fragmentation and power struggles among leadership (0-10) | 9.2 |
| external_intervention | float | FSI X1 indicator measuring foreign military, economic, or political interference (0-10) | 9.1 |
| target_conflict_next_year | float | Binary target variable: 1.0 if next year's fatalities exceed current year (escalation), 0.0 otherwise | 1.0 |

### Data Dictonary of Quantification and Uncertanity

| Table | Feature | Uncertainty | Source of Uncertainty |
|-------|---------|-------------|----------------------|
| fact_acled_event | fatalities | ± 20–30% for large events, often underreported | ACLED relies on media and partner reports; remote or censored areas have lower reporting rates; fatality counts are conservative "best estimates" |
| fact_acled_event | population_exposure | ± 10–25% depending on region | Derived from WorldPop population grids which are modeled estimates, not census counts; conflict zones have the weakest population data |
| fact_acled_event | centroid_latitude | ± 0.01–0.5° depending on geo-precision | ACLED codes events to nearest identifiable location; rural events may be coded to district or province centroid rather than exact site |
| fact_acled_event | centroid_longitude | ± 0.01–0.5° depending on geo-precision | Same as latitude; precision varies by event and source availability |
| fact_gdelt_event | goldsteinscale | ± 1–2 points per event | Score is assigned by CAMEO event code mapping, not human judgment; same real-world event can receive different scores depending on how the source article frames it |
| fact_gdelt_event | avgtone | ± 2–5 points per event | Computed by automated content analysis of source articles; sensitive to article language, translation quality, and source selection bias |
| fact_gdelt_event | nummentions | No fixed range; systematically biased upward for countries with heavier English-language media coverage | Countries like Israel and Turkey generate more media mentions per event than Yemen or Bahrain due to media access and international attention |
| fact_gdelt_event | numscores | Same media coverage bias as nummentions | Source count reflects media ecosystem size, not event severity |
| fact_country_year | acled_event_count | ± 5–15% depending on country | Underreporting in media-restricted zones (Syria, Yemen) means true event counts are likely higher than recorded |
| fact_country_year | acled_fatalities_sum | ± 20–40% at annual aggregation | Compounds per-event uncertainty across hundreds of events; countries with active censorship have the widest uncertainty bands |
| fact_country_year | acled_protest_sum | ± 10–20% | Protests in authoritarian states are underreported due to media suppression and self-censorship by journalists |
| fact_country_year | gdelt_avg_tone | ± 1–3 points at annual level | Averaging reduces per-event noise but systematic bias remains — countries covered primarily by state media vs international press will have different baseline tones |
| fact_country_year | gdelt_avg_goldstein | ± 0.5–1.5 points at annual level | Averaging smooths individual coding errors but CAMEO code assignment is deterministic, so systematic miscoding propagates |
| fact_country_year | gdelt_total_mentions | No fixed range; reflects media coverage volume, not conflict severity | Should be interpreted as a media attention proxy, not a direct conflict measure |
| fact_country_year | gdelt_total_scores | Same as nummentions | Source count proxy, not ground truth |
| fact_country_year | military_spending_current_usd | ± 5–15% for most countries; higher for conflict states | SIPRI notes that some countries do not report full military budgets; off-budget spending, corruption, and arms imports may not be captured; exchange rate fluctuations affect USD conversion |
| fact_country_year | fsi_total_points | ± 3–5 points (methodology-dependent) | Composite of 12 subjective indicators scored by content analysis software and expert review; methodology has been revised multiple times since 2006; year-to-year comparisons should account for scoring methodology changes |
| fact_country_year | demographic_pressures | ± 0.5–1.0 on 0–10 scale | Expert-coded indicator combining multiple data sources; subjective judgment involved in weighting |
| fact_country_year | refugees_idps | ± 0.5–1.0 on 0–10 scale | Based on UNHCR data which itself has reporting lags and coverage gaps in active conflict zones |
| fact_country_year | group_grievances | ± 0.5–1.5 on 0–10 scale | Most subjective FSI indicator; relies heavily on content analysis of media sources which may not capture ground-level communal tensions |
| fact_country_year | human_flight | ± 0.5–1.0 on 0–10 scale | Emigration data lags by 1–2 years in many countries; undocumented migration not captured |
| fact_country_year | economic_inequality | ± 0.5–1.0 on 0–10 scale | Gini coefficients and income data are sparse for conflict-affected economies |
| fact_country_year | economy | ± 0.5–1.0 on 0–10 scale | GDP estimates for conflict states like Syria and Yemen carry significant uncertainty; informal economies not captured |
| fact_country_year | state_legitimacy | ± 0.5–1.5 on 0–10 scale | Subjective assessment of government legitimacy; media framing affects scoring |
| fact_country_year | public_services | ± 0.5–1.0 on 0–10 scale | Service delivery data is unreliable in active conflict zones where infrastructure is destroyed |
| fact_country_year | human_rights | ± 0.5–1.0 on 0–10 scale | Based partly on Freedom House and press freedom indices which have their own methodological debates |
| fact_country_year | security_apparatus | ± 0.5–1.5 on 0–10 scale | Difficult to assess internal security dynamics from external sources; covert operations and intelligence activity not captured |
| fact_country_year | factionalized_elites | ± 0.5–1.5 on 0–10 scale | Political fragmentation is inherently difficult to quantify; relies on expert judgment about elite dynamics |
| fact_country_year | external_intervention | ± 0.5–1.5 on 0–10 scale | Covert foreign intervention (arms smuggling, intelligence support, proxy funding) is by definition underreported |
| fact_country_year | target_conflict_next_year | Binary, no continuous uncertainty | Derived deterministically from fatality comparison; inherits all uncertainty from the underlying acled_fatalities_sum values |






