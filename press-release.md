# New Conflict Hotspot Tool Could Save Lives in the Middle East

## Hook 
My dad served in Iraq during the War on Terror when I was a kid. Since then I've paid attention to the region; the cycles of violence, the displacement, and the way instability compounds over time. This project started from a simple question: if all this conflict data is sitting out there in the open, can we actually use it to see what's coming next?

## Problem Statement
Every year, armed conflicts across the Middle East displace millions of people, collapse economies, and cost countless lives. The ability to anticipate where violence will escalate before it happens could give policymakers, humanitarian organizations, and defense analysts the time they need to intervene and allocate resources. But most crisis response is still reactive — things blow up and then people scramble to respond.

Conflict early warning has traditionally relied on expert judgment. Analysts read reports, track diplomatic signals, and make informed guesses about where things might go wrong next. That approach doesn't scale. There are 15 countries in the Middle East alone, each shaped by a unique mix of political violence patterns, military posture, economic conditions, and state fragility. No team of analysts can process all of that simultaneously across every country, and expert predictions on geopolitical events have been shown to sometimes perform no better than chance.

The data needed to build something better already exists across multiple open sources — ACLED tracks conflict events in real time, GDELT monitors global media coverage and sentiment, SIPRI publishes military spending figures, and the Fund for Peace scores every country on state fragility — but these sources sit in isolation with different structures, formats, and granularities, making it difficult to combine them into a unified picture of escalation risk.

## Solution Description
This project brings those fragmented data sources together into a single relational dataset built specifically for conflict escalation prediction in the Middle East. Six interconnected tables covering 144,000+ ACLED conflict events, 2.1 million GDELT media events, military expenditure records, Fragile States Index scores, and ACLED's own Conflict Index, are joined through a central countries table and standardized to a country-year analytical grain.

The machine learning pipeline aggregates raw event data into annual features per country: total conflict events, fatality counts broken down by event type, media sentiment scores from GDELT, military spending trends, and all 12 FSI sub-indicators measuring things like security apparatus strength, factionalized elites, and external intervention. Classification models, including Random Forest and Gradient Boosting, then predict whether a country will experience elevated conflict in the following year relative to its own baseline.

The goal isn't to replace human analysts. It's to give them a tool that can flag emerging risk earlier and at a scale that manual monitoring can't match. When the model identifies a country whose conflict indicators are trending toward escalation, that's a signal for deeper investigation,not a final verdict.

## Chart
