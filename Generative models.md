# 5. Generative models 
_Lektion 9_

## Motivation


## Unsupervised learning
Lad os først lige få på det rene, at der er forskellige måder at arbejde på.
Når vi snakker om _Unsupervised Learning_, så snakker vi om at vi kun har data. Hvad betyder det? Jo, når vi træner et netværk til klassificering, så gør vi det på et dataset af billede, men der er også beskrivelser af de billeder. Så ved at træne for klassificering på datasættet det indeholder $(x, y)$, så søger vi at finde den funktioner der mapper $x$ til $y$.
Anderledes er det dog ved _Unsupervised Learning_, for her har vi som sagt ingen labels, vi har kun data; derfor er det vi søger i stedet, at finde en struktur i den data som vi har, hvor et klassisk eksempel på hvad man kan bruge _unsupervised learning_ på er clustering eller region proposals.

Og hey, en fordel er at vi ikke _skal_ bekymre os om de labels! Sæt der mangler nogle eller de er forkerte! Det ville ødelægge vores netværk, så lad os i stedet se på hvad vi kan gøre når vi ingen labels har, men kun ren, rå data!


- [x] Explain what unsupervised learning is, motivate its use, and compare it with supervised learning

## Models

Så nu har jeg jo godt nok trukket emnet, _Generative models_, og jeg er startet ud med at diskuttere måder hvorpå man kan lære på; derfor skal vi naturligvis også lige snakke om de modeller der _kan_ lære.

En model kan enten være:

> * Generative

Eller

> * Discriminative


Helt basalt, så de Generative modeler kan genere nye datapunkter og de discriminative, well, discriminate i mellem forskellige data punkter, som en klassificer gør.

> * Generative: laver ny
> * Discriminative: klassifiere

Hvis vi har et set af data X med labels Y, så vil en generativ model arbejde med den forenede sansynlighed $P(X, Y)$ hvis der findes en $Y$, ellers bare $P(X)$, og en discrimativ model arbejder med den betingeede sandsynligheden $P(Y | X)$.

> * Generative: laver ny, $P(X, Y)$
> * Discriminative: klassifiere, $P(Y | X)$

~~Joint probability is simply the likelihood that two events will happen at the same time, independently. That means the outcome of event X does not influence the outcome of event Y.~~

Lad os tage lidt forskellige eksempler. Hvis vi leger med en model der kan forudsige det næste ord i en sekvens, så er det typisk en generativ model, fordi den kan forudsige en sandsynlighed for en sådan sekvens. En discrimitativ model vel være ligeglad med spørgsmålet om noget er sansynligt, men bare fortælle hvor sansynligt det er.

Det er meget sværere at arbejde med og træne en generativ model end en discriminative model; lidt på samme må som det kræver en meget dybere forståelse af et materiale for en lærer til at kunne uddybe det så alle elever forstår det, end det er for en elev bare at se ligheden. Læreren skal kunne vende teorien rundt og op og ned, for at nogle elever kan se det fra en ny vinkel og forstå det på deres egen måde.

En discriminative model skal måske bare lære at se forskel på tallet “0” og tallet “1”, mens en generativ model skal have så dyb en forståelse for begge tal, at den skal kunne lave nye udgaver af dem.

- [x] Describe the difference between a generative model and a discriminative model in overall terms
- [x] Know that a generative model tries to learn the distribution of the data itself, and intuitively explain why this is a more challenging than train a discriminative model

## Autoencoder

15. Explain the architecture of a AE (including latent space representation and sampling layer)
4. Explain how a traditional autoencoder (AE) works and know the meaning of related terms, like encoder, decoder, and latent/hidden representation
5. Know that traditional AEs are typically used for dimensionality reduction and feature learning
6. Know which loss functions are typically used for training AEs
7. Know the difference between an undercomplete and overcomplete AE
8. Explain what a stacked AE is, and intuitively explain why it performs better than a one-layerAE
9. Know what latent interpolation is and mention examples of uses of it
10. Intuitively explain why traditional AEs tend to create gaps in the latent space representation
(makes the task easier for the decoder)
11. Mention regularization strategies that can be used in combination with AE
12. Explain what a Denoising AE is and how it works


### Convolutional autoencoders

13. Explain what a convolutional AE works and describe its overall architecture (especially upsampling techniques)


### Variational autoencoders

14. Explain conceptually how variational AEs differ from traditional AEs, and why they tend to learn smoother latent space representations
15. Motivate the use of the KL divergence term in variational AEs, and explain intuitively what the effect of the KL term is


## Generative Adversarial Networks

19. Explain how a GAN is trained (including definition of loss functions)
20. Know what a Deep Convolutional GAN (DCGAN) is and what the overall network architecture looks like
21. Know conceptually what a Conditional GAN is.
18. Explain what a Generative Adversarial Networks (GAN) does, and how it is designed (i.e., generator and discriminator)







17. Mention examples of latent space arithmetic

