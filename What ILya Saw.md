### In 2014, 2016, 2024, and 2023.

Dec 15, 2024

_LAST CALL for questions for our big 2024 recap episode! **[Submit questions and messages on Speakpipe here](https://www.speakpipe.com/LatentSpace) for a chance to appear on the show!** We record Monday._

---

It is remarkable how many AI professionals don’t take AI seriously — by which we mean fully thinking through the implications of scale and trendlines, and aligning their investments and actions accordingly with accelerating AI progress as the baseline. This is the main risk of the AI Engineer trying to plug current model capability gaps - trying to human-engineer their way out of something that should be or will soon be solved by machine-learning[1](https://www.latent.space/p/what-ilya-saw#footnote-1-153133301).

Properly establishing a mental framework/process for dealing with where ML training ends and AI engineering begins is of utmost interest to us. There is one person who many have credited for [feeling the AGI](https://futurism.com/openai-employees-say-firms-chief-scientist-has-been-making-strange-spiritual-claims) before anyone else: **Ilya Sutskever**.

This week we got not one but TWO insights into the insights of Ilya: his widely publicized Test of Time talk at NeurIPS ([transcript here](https://github.com/shun-liang/readable-talks-transcriptions/blob/main/neurips_2024/Vincent%20Weisser%20-%20.%40ilyasut%20full%20talk%20at%20neurips%202024%20pre-training%20as%20we%20know%20it%20will%20end%20and%20what%20comes%20next%20is%20superintelligence%20agentic%2C%20reasons%2C%20understands%20and%20is%20self%20aware.md)), and [OpenAI’s voluntary disclosure of his 2016 emails for the ongoing lawsuit with Elon](https://openai.com/index/elon-musk-wanted-an-openai-for-profit/). This gives us Ilya checkpoints for 2014, 2016, and 2024[2](https://www.latent.space/p/what-ilya-saw#footnote-2-153133301).
https://youtu.be/1yvBqasHLZs

We analyze each in time, ending with a final speculative conversation about 2023.

## What Ilya Saw in 2014

The relevant parts from his 2014 NIPS talk on sequence-to-sequence learning:

- **The Deep Learning Hypothesis** - “if you have a large neural network, it can do anything a human can do in a fraction of a second”
    
- **The Autoregression Hypothesis** - the simple next token prediction/sequence-to-sequence task would grasp the correct distribution to generalize from translation to everything else.
    
- **The Scaling Hypothesis** - “If you have a large big dataset, and you train a very big neural network, then success is guaranteed!”
    
- **The Connectionism Hypothesis** - If you believe that an artificial neuron is like a biological neuron[3](https://www.latent.space/p/what-ilya-saw#footnote-3-153133301), then very large neural networks can be “configured to do pretty much all the things that we human beings do”.
    

With this very selective interpretation of Ilya’s talk, discarding the couple things (LSTMs) that didn’t age well, Ilya has been consistently correct on the big, simple, yet profound insights. We would argue that generalizing insights that are correct on the big picture based on incorrect small details makes them more long-lived/credible. https://youtu.be/-uyXE7dY5H0

## What Ilya Saw in 2016-2017

OpenAI emails [published this week](https://openai.com/index/elon-musk-wanted-an-openai-for-profit/) also demonstrate deep beliefs from Ilya:

- **OpenAI was talent-constrained**: “_We can’t build AI today because we lack key ideas (computers may be too slow, too, but we can’t tell). Powerful ideas are produced by top people. Massive clusters help, and are very worth getting, but they play a less important role._”
    
    - **Advocates to 4x headcount**: “_Increase our headcount: from 55 (July 2017) to 80 (January 2018) to 120 (January 2019) to 200 (January 2020). We’ve learned how to organize our current team, and we’re now bottlenecked by number of smart people trying out ideas._” Greg Brockman later updates the end 2019 target to 300 and the end 2020 target to 900[4](https://www.latent.space/p/what-ilya-saw#footnote-4-153133301) .
        
- **Exploiting Parallelism and Networking leads to a super-Moore’s-Law kink in Hardware acceleration:**
    
    - _“There is good reason to believe that deep learning hardware will speed up 10x each year for the next four to five years. The world is used to the comparatively leisurely pace of Moore’s Law, and is not prepared for the drastic changes in capability this hardware acceleration will bring. This speedup will happen not because of smaller transistors or faster clock cycles; it will happen because like the brain, **neural networks are intrinsically parallelizable, and new highly parallel hardware is being built to exploit this**.”_
        
    - Nvidia would end up [buying Mellanox in 2019](https://en.wikipedia.org/wiki/Mellanox_Technologies).
        
- **One Big Experiment >> 100 small ones - and this means much bigger clusters**: _“95% of progress comes from the ability to run big experiments quickly. The utility of running many experiments is much less useful… Recently, it has become possible to combine 100s of GPUs and 100s of CPUs to run an experiment that’s 100x bigger than what is possible on a single machine while requiring comparable time. This has become possible due to the work of many different groups. As a result, the minimum necessary cluster for being competitive is now 10–100x larger than it was before. Currently, every Dota experiment uses 1000+ cores, and it is only for the small 1v1 variant, and on extremely small neural network policies. We will need more compute to just win the 1v1 variant. **To win the full 5v5 game, we will need to run fewer experiments, where each experiment is at least 1 order of magnitude larger (possibly more!)**. **TLDR:** What matters is the size and speed of our experiments. In the old days, a big cluster could not let anyone run a larger experiment quickly. Today, a big cluster lets us run a large experiment 100x faster.”_
    
    - **Advocates for large cluster:** “_Increase our GPU cluster from 600 GPUs to 5000 GPUs ASAP._”
        
    - We previously [talked with David Luan](https://www.latent.space/p/adepthttps://www.latent.space/p/adept), who was VP Eng at OpenAI who credited **OpenAI’s inclination to make big bets** vs Google’s (where he was also LLM lead) tendency to spread them.
        
        - Ilya: _“Over the past year, Google Brain produced impressive results because they have an order of magnitude or two more GPUs than anyone. We estimate that Brain has around 100k GPUs, FAIR has around 15–20k, and DeepMind allocates 50 per researcher on question asking, and rented 5k GPUs from Brain for AlphaGo. Apparently, when people run neural networks at Google Brain, it eats up everyone’s quotas at DeepMind.”_
            
    - We also [talked with Yi Tay](https://www.latent.space/p/yitay), author of UL2 and Reka Core who has now returned to [rejoin Noam Shazeer on DeepMind](https://www.yitay.net/blog/returning-to-google-deepmind), on **the rise of the Yolo run**
        
- **Self Playing Agents scale much better than human guidance**: _Self play in multiagent environments is magical: if you place agents into an environment, then no matter how smart (or not smart) they are, the environment will provide them with the exact level of challenge, which can be faced only by outsmarting the competition. So for example, if you have a group of children, they will find each other’s company to be challenging; likewise for a collection of super intelligences of comparable intelligence. So the “solution” to self-play is to become more and more intelligent, without bound. Self-play lets us get “something out of nothing.” The rules of a competitive game can be simple, but the best strategy for playing this game can be immensely complex._
    
    - Ilya wrote this in June 2017, 4 months before [AlphaGo Zero in October 2017](https://en.wikipedia.org/wiki/AlphaGo_Zero) demonstrated the power of self play in Go( scaling to 2000 Elo above [2015-2016’s AlphaGo](https://en.wikipedia.org/wiki/AlphaGo)), which then [generalized to AlphaZero](https://en.wikipedia.org/wiki/AlphaZero) two months later. Til this day [Noam Brown is still referencing the results](https://x.com/swyx/status/1867993162301510118/photo/3) of the self play exceeding frontier human performance, and is now [building a multi-agent team](https://x.com/swyx/status/1868008849183015344) after o1.
        
        [
        
        ![Image](https://substackcdn.com/image/fetch/w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Ff9bb7aed-1b99-4fd2-a1ee-0405916a3961_1078x1226.jpeg "Image")
        
        
        
        ](https://substackcdn.com/image/fetch/f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Ff9bb7aed-1b99-4fd2-a1ee-0405916a3961_1078x1226.jpeg)
        
    - _More on the 2023 section below…._
        
- **The importance of Curious AI**. “_How do we build curious systems?_” It is true that LLMs today are fundamentally uncurious, and that a sign of intelligence is an understanding that one does not know everything, and specifically one might not know everything needed to answer a question.
    
    - Elon has stolen this in the [Grok 2 system prompt](https://x.com/nyaathea/status/1867854474808811570?s=46): “`You are Grok 2, a curious AI built by xAI.”`
        

The [two](https://openai.com/index/openai-elon-musk/) prior [sets of published emails](https://www.techemails.com/p/elon-musk-and-openai) show:

- **how Ilya came around to closing off OpenAI for safety reasons**: “_The article is concerned with a hard takeoff scenario: if a hard takeoff occurs, and a safe AI is harder to build than an unsafe one, then by opensorucing everything, we make it easy for someone unscrupulous with access to overwhelming amount of hardware to build an unsafe AI, which will experience a hard takeoff. As we get closer to building AI, it will make sense to start being less open. **The Open in openAI means that everyone should benefit from the fruits of AI after its built, but it's totally OK to not share the science.**”_
    
- **how seriously Ilya takes AGI and that it ramped up over the OpenAI founding negotiations in 2017**: “_During this negotiation, we realized that we have allowed the idea of financial return 2-3 years down the line to drive our decisions. This is why we didn't push on the control — we thought that our equity is good enough, so why worry? But this attitude is wrong, just like **the attitude of AI experts who don't think that AI safety is an issue because they don't really believe that they'll build AGI**._”
    

## What Ilya Sees in 2024

**Pre-training as we know it will end.**

back to the NeurIPS 2024 talk:

[

![](https://substackcdn.com/image/fetch/w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Fd74c0d9b-b5b4-474a-b4da-9ea187307ee0_1188x988.png)



](https://substackcdn.com/image/fetch/f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Fd74c0d9b-b5b4-474a-b4da-9ea187307ee0_1188x988.png)

see also [comments on X](https://x.com/johnrushx/status/1867735273230282936)

- **The Age of Pretraining** was driven by exta-large neural networks on extra-large datasets.
    
    - but while compute is growing rapidly, data is not[5](https://www.latent.space/p/what-ilya-saw#footnote-5-153133301). Data is “the fossil fuel of AI” (we called it [low-background tokens](https://www.latent.space/p/nov-2023?open=false#%C2%A7the-concept-of-low-background-tokens)!)
        
    - what’s next? agents? synthetic data? inference time compute? Ilya was noncommittal on all of these.
        
        - However he did [work on GPT-Zero](https://www.reddit.com/r/singularity/comments/181oq4i/gptzerocould_this_be_what_q_relates_to/) in 2021, also calling it “test time compute”.
            
    - In biology, hominids broke the trendline of brain-body mass ratios vs primates and mammals. Biology has figured out different kinds of scaling in the past. So will AI research.
        
- **Superintelligence** will be:
    
    - “**agentic**”: current systems are only “very very slightly agentic”
        
    - “**reasoning**”: however, the more a system reasons, the more unpredictable it becomes. Chess AIs are unpredictable to the best human players.
        
    - “**understanding**”: from limited data, without getting “confused”.
        
    - “**self awareness**”: -we- are part of our own world models.
        

We of course know here that [Ilya raised $1b for “a straight shot to safe superintelligence”](https://decrypt.co/247897/safe-superintelligence-ilya-sutskevers-safe-ai-raises-1-billion) this year and that he thinks he has [identified what hill to climb](https://x.com/ilyasut/status/1831341857714119024?lang=en), but has given no hints on what it is beyond this chart at the talk:

[

![](https://substackcdn.com/image/fetch/w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Ff8db0bf5-70e0-4e5a-8490-5a787d513745_1184x1144.png)



](https://substackcdn.com/image/fetch/f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Ff8db0bf5-70e0-4e5a-8490-5a787d513745_1184x1144.png)

Ramin is also founder of Liquid AI, another [stealth-ish](https://x.com/swyx/status/1840794198913794236) nature-inspired new architecture startup - that just [raised $250m from AMD](https://x.com/LiquidAI_/status/1867597863053930730) this week. Another. nature-inspired lab is [Sakana which raised $100m in Sept](https://sakana.ai/series-a/).

## Bonus: What Did Ilya See in 2023?

We try as best we can to stay away from internal politics and kremlinology here, but when trying to understand OpenAI this is a particularly tough challenge. There are a few events that seem pertinent to us:

1. Ilya and Jan started [the Superalignment team](https://openai.com/index/introducing-superalignment/) in July, before what is now called [The Blip of Thanksgiving 2023](https://www.latent.space/p/the-end-of-openai-hegemony).
    
2. Nov 2023 demos of Q* (now Strawberry/o1) [impressed everyone](https://sfstandard.com/2023/11/17/openai-sam-altman-fired-apec-talk/): “_I can’t imagine anything more exciting to work on, and on a personal note, just in the last couple of weeks, I have gotten to be in the room, when we sort of like push the sort of the veil of ignorance back and the frontier of discovery forward and getting to do that is like a professional honor of a lifetime._”
    
3. The team then published [Weak-to-strong generalization](https://openai.com/index/weak-to-strong-generalization/), which takes some lessons from self play Ilya saw in 2017: _“Current alignment methods, such as reinforcement learning from human feedback (RLHF), rely on human supervision. **However, future AI systems will be capable of extremely complex and creative behaviors that will make it hard for humans to reliably supervise them.”**_
    
4. The entire Superalignment [team has now left](https://www.vox.com/future-perfect/2024/5/17/24158403/openai-resignations-ai-safety-ilya-sutskever-jan-leike-artificial-intelligence), including some [departures](https://x.com/RosieCampbell/status/1863017727063113803) on the [Policy](https://x.com/Miles_Brundage/status/1849138802864087234) and [Governance](https://x.com/richardmcngo/status/1856843040427839804?s=46) team, and [alleged leakers](https://www.theinformation.com/articles/openai-researchers-including-ally-of-sutskever-fired-for-alleged-leaking?rc=ytp67n) who then published [a larger manifesto raising situational awareness](https://situational-awareness.ai/) on the Intelligence Explosion, race to large clusters, and need for security.
    

o1 has now been released in the wild for 2 months (for -mini and -preview) and a week (for pro) and the world has not yet ended. People like [Dylan Field](https://x.com/zoink/status/1867703768147697990) and [employees](https://x.com/VahidK/status/1865140156812136924) are hyping it as AGI, but it is costly and slow. The [o1 system card](https://openai.com/index/openai-o1-system-card/) shows standard, minor mitigation effects. Did Ilya overestimate o1, underestimate OpenAI’s ability to mitigate negative effects, or is the world about to end?

[

![](https://substackcdn.com/image/fetch/w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Fa71745bf-a856-42f8-a280-90969effd6ec_1490x956.png)



](https://substackcdn.com/image/fetch/f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Fa71745bf-a856-42f8-a280-90969effd6ec_1490x956.png)

[

![](https://substackcdn.com/image/fetch/w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Fac708b43-b2c6-4862-8e85-1c89f64b1799_1490x956.png)



](https://substackcdn.com/image/fetch/f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Fac708b43-b2c6-4862-8e85-1c89f64b1799_1490x956.png)

[1](https://www.latent.space/p/what-ilya-saw#footnote-anchor-1-153133301) Nice example in [our recent realtime API post on the move from chained models to omnimodel](https://www.latent.space/p/realtime-api).

[2](https://www.latent.space/p/what-ilya-saw#footnote-anchor-2-153133301) He has done other talks in the interim, eg [on Generalization](https://sumanthrh.com/post/notes-on-generalization/).

[3](https://www.latent.space/p/what-ilya-saw#footnote-anchor-3-153133301) [This is questioned](https://x.com/mkieffer1107/status/1867706489873523014) by neuroscientists.

[4](https://www.latent.space/p/what-ilya-saw#footnote-anchor-4-153133301) Note [Michelle Pokrass’ comments](https://www.latent.space/p/openai-api-and-o1) on the growth of the Applied AI team

[5](https://www.latent.space/p/what-ilya-saw#footnote-anchor-5-153133301) Many people [disagree](https://x.com/kalomaze/status/1868015615723917624?s=46)!