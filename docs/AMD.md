1. Accumulation – what the literature actually agrees on
1.1 Core idea (all frameworks)
Across Wyckoff, general TA, and AMD/ICT content, “accumulation” is consistently described as:
Happens after a downtrend


Sideways, range-bound trading for some time


Smart money gradually buying, but doing it in a way that doesn’t move price too much


Often looks like “boring chop” to retail, but it’s actually large players absorbing sell orders from weak hands (Managed Accounts IR)


Wyckoff and Wyckoff-style guides call this the accumulation trading range: a horizontal zone where price moves sideways while big players load up. (Alchemy Markets)
In AMD/ICT explanations, the same concept shows up as:
Asian session = accumulation: low volatility, consolidation around the daily open, building positions before London/NY move. (TradingFinder)


So the high-level story from reputable-ish sources is consistent.

1.2 Wyckoff’s detailed accumulation structure (swing / higher timeframe)
Even though you’re doing intraday FX, Wyckoff is still the cleanest “gold standard” description of accumulation. Summarizing multiple modern Wyckoff explainers:
Context:
A prior downtrend with increasing selling pressure. (TradingFinder)


Key price & volume events in accumulation:
Preliminary Support (PS)


After a long decline, you see strong down bars with larger volume where price starts to react up from a support area.


Shows that big buyers are starting to absorb supply, but the downtrend isn’t fully finished. (Power Trading Group)


Selling Climax (SC)


Panic sell-off, wide bearish candle(s), volume spike, price hits a major low.


Often marks the point where retail capitulates and smart money absorbs a lot of that selling. (Power Trading Group)


Automatic Rally (AR)


When the panic selling dries up, price bounces sharply up on reduced selling.


The high of this rally + the SC low typically define the boundaries of the new range. (Power Trading Group)


Secondary Tests (ST)


Price returns to the SC area, but this time down moves have smaller spread and lower volume.


Confirms that supply is drying up; sellers hitting the lows are getting absorbed. (Power Trading Group)


Spring / Shakeout (not always present)


A fake break below the range low, usually with a quick recovery back into the range.


Wyckoff explicitly sees this as a form of manipulation inside accumulation – grabbing liquidity and shaking out the last weak holders. (Alchemy Markets)


Sign of Strength (SOS) & Jump Across the Creek (JAC)


Strong rallies out of the range on rising volume; pullbacks are shallow and on low volume.


This transitions the market from accumulation into markup (the new uptrend). (Alchemy Markets)


From this, you can see accumulation is not just “flat market” – it’s a narrative in price + volume:
Selling pressure climaxes,


Then gets tested multiple times,


Then you often get one last fake breakdown (spring),


Then genuine strength appears.



1.3 How general FX/TA sources describe accumulation
FX-specific and general TA sources mostly echo Wyckoff but in simpler language:
Sideways after a downtrend: “A prolonged period of consolidation following a downtrend” where big players accumulate at low prices. (Managed Accounts IR)


Low net progress but active participation:


Price seems “stuck” in a horizontal band.


Volume can increase even though price doesn’t move much → demand absorbing supply. (The Forex Geek)


Absorption of sell orders:


One FX thread explicitly:


 “Accumulation is where the market is absorbing sell orders to push prices higher” (Forex Factory)



So from a research standpoint, reliable sources agree that:
Accumulation = sideways range after a decline, with evidence that selling is being absorbed (volume/behavior at lows), not rewarded.
That’s the essence your bot needs to detect.

1.4 Intraday / AMD / ICT view on accumulation (especially Asian session)
Now zoom into the intraday, session-based version, which is what you’re actually going to code.
Several AMD / ICT / SMC style resources describe:
Asian session as the daily accumulation phase:


Typically low volatility, smaller ranges vs London/NY.


Often a consolidation around the daily open where smart money builds positions. (TradingFinder)


ICT “Power of 3” / AMD:


Accumulation = when the day opens and price ranges near the open, “trapped” in a horizontal range.


Smart money accumulates positions while retail buys support/sells resistance inside that box. (HowToTrade)


Many ICT derivatives and tools explicitly say:


 “Asian Session acts as the accumulation phase before the real move” (Forex Factory)



So for your bot’s context (forex day trading, AMD/PO3 style), “accumulation” will mostly mean:
Context: early session, often Asia


Behavior:


Narrow, stable range (small compared to recent daily ranges/ADR)


Price oscillating around daily open or a nearby level


No sustained trend, wicks both sides, liquidity building above and below the range


We’ll later translate that into numeric thresholds.

2. Manipulation – how it’s described
2.1 Concept across Wyckoff & AMD/ICT
From Wyckoff-ish and SMC/AMD sources:
Manipulation = engineered false move to take liquidity, not just any random spike.


In Wyckoff language:


Spring (after accumulation) and upthrust (after distribution) are classic manipulative moves:


Price breaks through support or resistance into stops,


Then snaps back into the range, exposing it as a trap. (Alchemy Markets)


In AMD / SMC / ICT world:


It’s often called:


Liquidity grab / liquidity sweep,


Stop hunt,


Judas swing. (studylib.net)


Description:


Market runs above Asian high or below Asian low at London or NY open,


Takes out resting orders and stop-losses,


Then reverses in the opposite direction (the “real move”).


One FX forum summarized it neatly:
“… Manipulation is when the market throws a fake breakout from these zones. Wyckoff calls this a ‘spring’.” (Forex Factory)

2.2 Practical characteristics of manipulation (price behavior)
From AMD/ICT and broker education articles on AMD: (XS)
Common features they highlight:
Break of a known level, then quick failure


The known level:


Asian range high/low, previous day’s high/low, key session high/low, major support/resistance.


Price pierces it, triggers orders, then quickly returns back.


Volatility spike


Candles are larger than recent average (range expansion) during manipulation.


Often around session opens (London open, NY open).


No sustained follow-through in direction of the break


After the spike, price fails to continue in that direction and instead runs the other way.


Volume confirmation (where volume is available)


Often a sharp volume spike as stops and breakout orders get filled.


Subsequent move in the opposite direction may show strong volume, indicating smart money taking the other side.


So, manipulation is basically:
A fake breakout (often at session open) beyond a liquidity pool, that fails and reverses.

3. Distribution – opposite side of the same coin
3.1 Wyckoff view
Wyckoff describes distribution as:
The mirror image of accumulation, but at high prices after an uptrend.


Institutions offload long positions to late buyers over time in a range. (QuantVPS)


Key distribution events:
Preliminary Supply (PSY): first sign of large selling after a strong uptrend, with high volume and wide spreads.


Buying Climax (BC): a final, aggressive push up on huge volume, followed by sharp reaction down.


Automatic Reaction (AR): the first fast drop after BC.


Upthrusts / UTAD: false breakouts above the range highs, similar to springs but at the top. (Power Trading Group)


After that, supply dominates and price moves into markdown (downtrend).
3.2 General / FX descriptions
Broker and education sites describe distribution as: (The Forex Geek)
Range after a strong uptrend, usually near new highs.


Smart money selling into strength, while:


Retail sees bullish headlines/new highs and keeps buying.


Often accompanied by:


Increasing volume near the top,


Failed attempts to make new highs,


Rising wicks above the range, then eventual breakdown.


In AMD articles:
Accumulation → Manipulation → Distribution is framed as:


Accumulation at lower prices,


Manipulative false moves to load more / trap retail,


Distribution as the phase where institutions unload into retail euphoria. (XS)


So for your bot:
Distribution = range or topping behavior at/near the high after an up-move, with signs that up moves are being sold (upthrusts, failed breakouts, heavier volume on down legs).

4. What’s common across the “schools”
You basically have three overlapping “languages” describing the same thing:
Wyckoff / classical tape-reading


Modern AMD / Smart Money Concepts / ICT PO3


Generic FX education & broker articles


The overlap:
Accumulation


After a decline,


Sideways range,


Smart money absorbing opposite orders (usually selling),


Low net movement, but evidence of demand absorbing supply.


Manipulation


Fake breakout beyond a well-watched level (range high/low, seen liquidity),


Stop hunt / liquidity grab,


Often concentrated around session opens in intraday FX,


Followed by a sharp reversal back into or through the range.


Distribution


After a rise,


Sideways / topping range near highs,


Smart money selling into late buying,


Often includes false breakouts (upthrusts) before a bigger drop.


Session-based AMD/PO3 (Asia → London → NY) basically compresses Wyckoff’s swing-timeframe cycle into a single day. (Rattibha)

5. Draft checklists (research version, not final code)
Here’s a research-style checklist you can refine, backtest, and only later turn into strict numeric rules.
5.1 Accumulation checklist (priority)
From all the sources:
Context
✅ Prior downtrend (lower lows / lower highs, or big decline over X sessions/days). (Managed Accounts IR)


Structure (price)
✅ Price enters a horizontal range – clearly defined support & resistance.


✅ Range appears after the main selling leg, not before it. (Managed Accounts IR)


Volatility
✅ Session/daily range smaller than recent average (lower volatility).


✅ Candles frequently overlap and lack long directional sequences. (TradingFinder)


Volume / orderflow (if available)
✅ High volume at the lows (SC, ST) followed by reduced volume on further pushes down (selling is being absorbed / fading).


✅ Rising volume on rallies, declining volume on pullbacks inside the range. (Power Trading Group)


Springs / fake breakdowns (advanced but powerful)
✅ Occasional break below support that quickly returns into the range (spring).


✅ Those breaks often show a volume spike but fail to follow through down. (Alchemy Markets)


Intraday Asian-session flavor
✅ Asian session shows narrow range, low volatility vs previous days / London & NY.


✅ Price often oscillates around the daily open or midline.


✅ High & low of Asia later act as liquidity targets for London/NY. (TradingFinder)


If you want your bot to be conservative, you can require most of these to be present before calling it “accumulation”.

5.2 Manipulation checklist
Location
✅ Occurs around clear levels:


Asian high/low


Yesterday’s H/L


Range boundary from accumulation / distribution. (studylib.net)


Timing
✅ Often near session opens:


London open for Asian range sweeps


NY open for London range sweeps. (Time Price Research)


Price behavior
✅ Fast move through the level (stop run), usually with:


Candle range larger than recent candles


Or strong momentum push


✅ Lack of continuation:


After the break, price fails to progress in that direction


Quickly returns inside the previous range or reverses strongly. (XS)


Volume (if visible)
✅ Spike in volume at the breakout, followed by a reversal.


This matches both Wyckoff’s “spring/upthrust” and AMD’s liquidity-grab explanations. (LuxAlgo)



5.3 Distribution checklist
Context
✅ Strong prior uptrend / markup. (QuantVPS)


Structure
✅ Price starts to form a horizontal or slightly rounding range near the top.


✅ Multiple tests of resistance with:


New highs failing to follow through (wicks, quick rejections). (Power Trading Group)


Volume / orderflow
✅ High or rising volume near the top, especially on up moves that fail.


✅ Sometimes lower volume on final pushes up, showing buyers are exhausted while pros sell. (Alchemy Markets)


Manipulative events
✅ Upthrusts – brief pushes above range high or previous swing high that quickly fail.


These are the “springs at the top” and are part of both Wyckoff and AMD/SMC descriptions. (Alchemy Markets)


Intraday AMD context then maps this to:
Asia accumulation → London manipulation → NY distribution
 where NY is often the trend completion / exit phase, unloading positions as retail finally jumps on. (XS)



6. How this helps us design the bot later (without gambling)
For now, your takeaway isn’t “here’s the formula,” but:
We now have a sourced, consistent definition of:


What accumulation looks like structurally and behaviorally,


What manipulation is (in both Wyckoff and AMD/ICT terms),


What distribution is and where it appears in the cycle.


When you’re ready to move on from research to design, we can:
Pick a base timeframe (e.g. 15m or 5m),


Turn these checklists into explicit measurable conditions:


e.g. “session range < X × ADR”, “number of overlapping bars > Y”, etc.,


And require multiple conditions to be true before the bot labels a phase as accumulation / manipulation / distribution.


That way your bot isn’t “predicting”; it’s classifying based on behaviors that are well-documented across Wyckoff, SMC, and ICT material.
Nice, this helps a lot because now I see exactly what your engine looks like under the hood 👀
You already have a solid breakout framework. We don’t need to rewrite it — we just need to wrap it in AMD context so:
It only fires in the “right phase” of the day, and


It respects what Asia + London structure is saying.


And we can still stay in the “no rush to code” mode: I’ll talk architecture + logic, not implementation details yet.

1. What your current bot is actually doing (in plain language)
Your PurePriceActionStrategy essentially:
Finds swing highs/lows using a swing_window.


Builds support/resistance levels from the most recent swings.


Detects breakouts:


Bullish: close > resistance + threshold


Bearish: close < support – threshold


Confirms with candlestick patterns (optional).


Uses ATR + buffers to place SL, then RR for TP.


Applies filters:


Spread guard


Duplicate breakout suppression


Headroom to next obstacle (needs enough room to TP)


So it’s a clean breakout engine with good risk filters — but it’s blind to time-of-day and AMD structure. It sees a breakout the same way at:
02:00 (Asia chop)


08:00 (London open)


18:00 (NY late session)


From an AMD mindset, that’s too “context-blind”.

2. Idea: AMD as a context gate around this breakout bot
Instead of making an “AMD bot”, think:
“AMD context decides when this breakout engine is allowed to trade, and in which direction.”
So you keep:
All the breakout logic,


All the ATR/SL/TP logic,


And add a thin AMD filter layer that can do things like:
“It’s Asian accumulation, so don’t take big breakouts yet.”


“We’re in London manipulation phase, so be careful of fake breakouts beyond Asia H/L.”


“We’re after a clean manipulation sweep, now allow breakouts in the expansion direction.”


Architecturally, that fits right at the start of generate_signal.

3. Where AMD plugs into this code (conceptual)
Your key entry point:
def generate_signal(self, data: pd.DataFrame, symbol: str, mtf_context: Optional[dict] = None) -> Optional[TradingSignal]:

There are two straightforward ways to inject AMD context:
Option A – Add an amd_context argument
Like:
def generate_signal(self, data, symbol, mtf_context=None, amd_context=None):
    # early AMD filters here

Where amd_context is a simple dict or object such as:
amd_context = {
    "session": "LONDON",            # "ASIA", "LONDON", "NY"
    "asia_type": "ACCUMULATION",    # "ACCUMULATION", "EXPANSION", "UNKNOWN"
    "phase": "MANIPULATION",        # expected current phase, optional
    "asia_high": 1.0850,
    "asia_low": 1.0800,
    "is_after_sweep_of_asia_low": True,   # etc.
}

Then before you compute swings / breakout, you can say:
If session == "ASIA" and you don’t want to trade Asia → return None.


If asia_type == "UNKNOWN" and your AMD model says “skip this day” → return None.


If you’re in London but haven’t seen the expected manipulation yet → maybe skip certain breakouts.


Option B – Let AMD live outside and decide if it even calls generate_signal
In other words:
A higher-level controller computes AMD state for the day/session.


If AMD conditions are satisfied, it calls generate_signal.


If not, it doesn’t call it at all.


This keeps PurePriceActionStrategy completely generic and AMD-free, but AMD still controls when it’s used.
Both are valid. For a neat architecture:
AMD module = “what type of day/session is this?”


Breakout module = “given I’m allowed to trade, is there a breakout worth taking?”



4. Concrete ways AMD can improve this breakout strategy
Let’s tie this back to your “when do we trade?” question.
4.1 Only trade breakouts that match the AMD phase
Example for Asia accumulation → London manipulation → NY distribution:
During Asia


You may choose:


No trading, or


Only very small size or range-style scalps (but with a different, dedicated strategy).


For the breakout bot, you probably want:


if session == "ASIA": return None


London manipulation phase
 The danger here: false breakouts through your support/resistance.

 AMD filter can say:


If price is breaking through Asia high/low for the first time near London open
 → treat that as potential manipulation, not a clean expansion yet.


So:


Either block the first breakout beyond Asia H/L and wait for a retest/reclaim, or


Require extra confirmation (e.g. BOS + FVG in opposite direction) before allowing trades.


So if amd_context["is_first_sweep_of_asia_high"] and it’s right at London open, you might:


Block trend breakout in the sweep direction (because it might be manipulation).


Later, once price comes back inside and breaks out the other way, then allow your breakout bot to fire.


Post-manipulation expansion (still London, maybe into NY)
 This is where your breakout bot is perfect:


After the sweep + reversal is confirmed, you want:


Breakouts in the direction of expansion


Past new swing levels, with clean structure


Here AMD filter can enforce directional bias. For a bullish day:


allowed_direction = "BUY"


So if BreakoutInfo.type == 'bearish', you can just: return None.


That stops you shorting pullbacks in what AMD sees as an expansion leg.


New York distribution / late stage

 Once AMD module marks “we’re in distribution / late trend” you can:


Disallow new breakouts in trend direction (momentum is fading),


Or only allow:


Tighter RR, or


Scaling out, not new entries.


So AMD can say:

 if amd_context["phase"] == "DISTRIBUTION":
    # no new breakout entries in trend direction
    if bo.type == amd_context["trend_direction"]:
        return None
 Again, conceptually — we’ll define these precisely later.



4.2 Use Asia range as special S/R inside breakout logic
Right now, you build S/R from swing points only. With AMD, you can tell the breakout bot:
“Asian High and Asian Low are always S/R of interest.”
So:
When computing resistance / support, you can inject asia_high and asia_low (if not already present).


Then _detect_breakout_close will see those as breakout levels.


But AMD context will decide if those breakouts are valid or likely manipulation:


Example:


First break of Asia low at London → treat with suspicion.


Second break after failed sweep → more likely genuine.


This marries AMD’s liquidity narrative with your existing swing-based breakout math.

4.3 Time-of-day filters
Your current code doesn’t care what time it is.
With AMD, you can add a simple time/session filter:
Session classification (based on UTC hour):


Asia, London, New York.


Config options like:


enable_asian_trading: bool


enable_london_trading: bool


enable_ny_trading: bool


Then, before anything:
if amd_context["session"] == "ASIA" and not config.enable_asian_trading:
    return None

This alone often improves performance: most breakout systems perform very differently by session.

5. What we should do before coding any of this
You said you don’t want to rush into code (good 🧠). Here’s the order I’d follow:
Define AMD context object (on paper):


What fields does it have?


session, asia_type, trend_direction, has_swept_asia_high, etc.


Which of those fields do you want to use as filters for the breakout strategy?


Decide “allowed trades” by scenario:


Asia = Accumulation, London = Manipulation day:


What’s allowed in London? In NY?


Asia = Expansion:


Do you allow countertrend breakouts? Under what conditions?


Write plain-language rules like:


“If Asia accumulates and London sweeps Asia low and reclaims the box, only then allow bullish breakouts above [X] within Y bars of the reclaim.”


Only then translate those into:


Additional parameters / context to generate_signal,


A few early-return filters and directional checks around your existing breakout logic.


That way your AMD integration is:
Clean,


Testable,


And doesn’t mutate the core breakout behavior — it just filters it.




