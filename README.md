# SkillCorner X PySport Analytics Cup
This repository contains the submission template for the SkillCorner X PySport Analytics Cup **Research Track**. 
Your submission for the **Research Track** should be on the `main` branch of your own fork of this repository.

Find the Analytics Cup [**dataset**](https://github.com/SkillCorner/opendata/tree/master/data) and [**tutorials**](https://github.com/SkillCorner/opendata/tree/master/resources) on the [**SkillCorner Open Data Repository**](https://github.com/SkillCorner/opendata).

## Submitting
Make sure your `main` branch contains:
1. A single Jupyter Notebook in the root of this repository called `submission.ipynb`
    - This Juypter Notebook can not contain more than 2000 words.
    - All other code should also be contained in this repository, but should be imported into the notebook from the `src` folder.
2. An abstract of maximum 500 words that follows the **Research Track Abstract Template**.
    - The abstract can contain a maximum of 2 figures, 2 tables or 1 figure and 1 table.
3. Submit your GitHub repository on the [Analytics Cup Pretalx page](https://pretalx.pysport.org)

Finally:
- Make sure your GitHub repository does **not** contain big data files. The tracking data should be loaded directly from the [Analytics Cup Data GitHub Repository](https://github.com/SkillCorner/opendata).For more information on how to load the data directly from GitHub please see this [Jupyter Notebook](https://github.com/SkillCorner/opendata/blob/master/resources/getting-started-skc-tracking-kloppy.ipynb).
- Make sure the `submission.ipynb` notebook runs on a clean environment.

_⚠️ Not adhering to these submission rules and the [**Analytics Cup Rules**](https://pysport.org/analytics-cup/rules) may result in a point deduction or disqualification._

---

## Repo running guide

This repository only uses simple data analysis libraries like pandas, numpy, json and existing sports analytics libraries like kloppy, databallpy, mplsoccer etc.
However, ONLY IF THESE MODULES ARE ABSENT, please use pip install -r requirements.txt.

# Proximity Score: A New Metric and its Effect on Expected Threat Outcomes

## Research Track Abstract (max. 500 words)

# Introduction
Quantifying defensive pressure is crucial for understanding how teams disrupt buildup and suppress chance creation. However, most existing approaches measure pressure only on the **ball carrier**, overlooking how defenders also constrain **passing options**, which strongly influence attacking progression.

This work introduces **Proximity Score**, a tracking‑derived metric that captures how close defenders are to both the player in possession and all viable receiving options during each phase of play. Using the SkillCorner × PySport Analytics Cup dataset—which includes player tracking, dynamic events, and Phases of Play segmentation—I analyse how defender proximity relates to changes in **expected threat (xThreat)**, possession outcomes, and decision-making under pressure. The aim is to identify when and where defensive proximity most effectively reduces attacking danger.

# Methods
### Data Sources
The analysis combines:
- **Tracking data** (player positions and detection flags)  
- **Dynamic events** (possessions, passing options, off‑ball runs, engagements)  
- **Phases of Play**, which define tactical contexts such as Build‑Up, Create, and Finish  

All data is loaded directly from the SkillCorner Open Data repository.

### Event Enrichment & xThreat Metrics
Each player possession is linked to associated passing options and on‑ball engagements. For each possession:
- **xThreat at start** is inherited from the previous successful pass.
- **xThreat at end** corresponds to the outgoing pass.
- **xThreat potential** is the maximum of *(xThreat × pass completion probability)* among all passing options.

From these values:
- **xThreat Increase** = end − start  
- **Potential xThreat Reduction** = potential − end  

### Proximity Score
Proximity reflects instantaneous defensive pressure.

For each passing option or possession event, at every frame *t*:
1. Identify all defenders.
2. Compute distance from event player to each defender.
3. Take the nearest-defender distance.

The **Proximity Score** is the mean across all frames of the event:


$$
\text{Proximity}(E) = \frac{1}{T}\sum_{t=1}^{T}
\min_{d \in \text{defenders}} \lVert p_{E}(t) - p_{d}(t) \rVert
$$



This yields:
- **PP Proximity Score:** pressure on the ball carrier  
- **PO Proximity Score:** pressure on each passing option  
- **Average PO Proximity:** mean proximity across all linked passing options  

### Phase Aggregation
All possessions within each phase window are extracted, and their proximity and xThreat metrics aggregated to evaluate how pressure influences threat creation and phase transitions.

# Results
Visual analyses comparing xThreat Increase and Potential xThreat Reduction (shown here) against possession and passing‑option proximity revealed a consistent pattern:  
**lower proximity (i.e., tighter defensive pressure) leads to smaller xThreat increases and larger unrealised threat.**

![alt text](https://github.com/falguni7/analytics_cup_research_fg/blob/main/abstract_fig/fig_pot_xt_reduction.png)

Under high pressure, attackers more frequently select passing options with lower xThreat and lower completion probability, even when better options exist. This effect is most pronounced in Build‑Up and Create phases, where defenders are compact and decision time is limited.

# Conclusion
Defensive proximity meaningfully reduces attacking efficiency. When defenders are close to both the ball carrier and passing options, teams are more often forced into suboptimal choices, yielding lower realised threat and a larger gap between potential and actual danger.  

Proximity Score therefore offers a simple, interpretable way to quantify pressure on the entire attacking structure—not just the possessor—providing a foundation for analysing pressing styles, identifying effective defenders, and understanding how defensive positioning shapes possession outcomes. With higher amount of league-wide tracking data, ML models can be employed to identify different teams' pressure and structure patterns and xthreat effects for and against.
