# Under Virtual Skies

by jbk, sebastian and bolli featuring Lena and WORKINPROGRESS at Driebergen-Rijsenburg Summer School 2023.

[![Under Virtual Skies Music Video](https://img.youtube.com/vi/7R_EcUjyeU0/0.jpg)](https://www.youtube.com/watch?v=7R_EcUjyeU0)

We are a collective of individuals who crossed paths during the "Digital Europe" summer school in Driebergen, Netherlands, while participating in the "AI in Creative Processes" working group. Our shared mission was to craft a memorable AI-generated hit song for the summer school, ensuring that every participant had a delightful melody to cherish. Within our team, jbr and bolli orchestrated the song through an AI co-creation process, while matthias, julia, djamila, yann, and michael collaborated to produce the accompanying music video.

The song encapsulates our journey during the "Digital Europe" summer school held in Driebergen, Netherlands, throughout August 2023. This event, jointly organized by the German Scholarship Foundation (Studienstiftung des deutschen Volks), the College d'Europe, and the Austrian and Swiss Scholarship Foundations, was a remarkable experience. Our aim was to preserve the multitude of impressions and joyful moments from this summer school, ensuring they remain etched in our memories long after the event's conclusion.

Our Human-AI Co-Creation began with an exploration of music generation tools, notably MusicGen from the AudioCraft library. Initially, we were fascinated by the outcomes, but soon realized that the results sounded repetitive. The AI struggled to craft compelling melodies or coherent structures independently. Frustration crept in as our efforts to enhance MusicGen's results often led to an unwelcome surplus of noise, leading to a sense of limitation.

In response, we transitioned to Google's Magenta tools, which allowed us to generate MIDI files. With Magenta, we produced a multitude of MIDI melodies, sifting through them to uncover the gems. While many melodies fell by the wayside, we eventually unearthed compositions that resonated with us.

Our creative journey extended to generating lyrics with ChatGPT, chosen for its availability. Similar to our experiences with music generation, we discovered that AI couldn't autonomously conjure structure or meaning, including rhymes. However, ChatGPT demonstrated proficiency in inspiring our creative process by providing valuable ideas and metaphorical expressions. We selectively incorporated the most compelling lyrical elements, arranging them to form a cohesive narrative.

From that point onward, the creative process was primarily driven by our human touch. We lent our voices to the vocals and fine-tuned the timbre of the MIDI instruments, breathing life into our collaborative creation.

To create the video, we let an AI "dream". We adapted an implementation of the deep dream algorithm. Instead of adjusting weights in a neural network to match an output, the method takes the same calculations but asks what the network "sees" on a particular layer and how the image must be changed to reinforce this vision. Furthermore, we explored how different AI image generation tools, for instance stable diffusion, can imitate the style of famous artist like  van Gogh. The results are integrated as a slideshow in the deep dream algorithm. An image is shown for a second before "dreamed away" by the AI. To create a cyber-space-like experience, we also implemented a possiblitity to spawn smaller images at random positions. As desserts played an important role for us on our AI exploration event, they are used for those popup effects. Of course, they are also AI generated. At the end, we show some photos of us working on the project.

Make sure to check out the [other projects](../index.md) from our working group "AI in Creative Processes"!
