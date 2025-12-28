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

## Research Track Abstract Template (max. 500 words)

# Introduction

Quantifying defensive pressure is essential for understanding how teams disrupt opposition buildup and suppress chance creation, yet most existing models only measure pressure on the **ball carrier**. This overlooks the fact that attacking progression often depends on the positioning and availability of **passing options**, not just the current possessor.  

In this work, I propose **Proximity Score**, a tracking‑derived metric that measures how close defenders are to both the player in possession and all viable receiving options during each phase of play.  

Using the SkillCorner × PySport Analytics Cup dataset—which provides dynamic events, player tracking, and Phases of Play segmentation—I investigate how defender proximity influences changes in **expected threat (xThreat)**. The goal is to identify where and when defensive pressure is most effective, and how pressing behaviour varies across pitch zones and tactical phases.


# Methods

### Data Sources
The analysis uses:
- **Tracking data** for player positions and movement.
- **Dynamic events** (player possessions, passing options, off‑ball runs, on‑ball engagements).
- **Phases of Play**, which define tactical contexts such as Build‑Up, Create, and Finish.

All data is loaded directly from the SkillCorner Open Data repository using the provided URLs and tools.

### Event Linking & Threat Metrics
Each player possession is enriched by:
1. Linking associated passing options and related child events.  
2. Computing:
   - **xThreat at start**: inherited from the previous possession’s successful pass.
   - **xThreat at end**: based on the outgoing pass.
   - **xThreat potential**: the maximum *(xThreat × pass completion probability)* among all passing options.

From these we derive:
- **xThreat Increase** — realised attacking progression.  
- **Potential xThreat Reduction** — danger prevented by effective defensive pressure.

### Proximity Score
For each frame in a possession or passing option:
- Identify defenders (opposing team players).
- Compute the nearest‑defender distance.
- Average these values across the event duration.

This yields:
- **PP Proximity Score** (pressure on the ball carrier)  
- **PO Proximity Score** (pressure on each passing option)  
- **Average PO Pressure per possession**

### Phase‑Level Aggregation
Using phase frame windows, all possessions within each phase are extracted. Their proximity and xThreat metrics are summarised to evaluate how defensive pressure affects attacking threat, possession outcomes, and phase transitions.

# Results

# Conclusion
