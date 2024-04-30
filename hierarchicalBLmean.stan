data {
  int<lower=0> nSubjects;
  int<lower=0> nTrials;
  int<lower=0> nBandits;
  int<lower=0> shock[nSubjects,nTrials];
  int<lower=0> option_A[nSubjects,nTrials];
  int<lower=0> option_B[nSubjects,nTrials];
  int<lower=0> choice_stim[nSubjects,nTrials];
  real<lower=0> payoff[nBandits];
}

transformed data {
  vector[nBandits] initV; //initialise monetary payoff for fractals. This is assumed known and not subjected to learning.
  vector[nBandits] initALPHA; //initialise alpha & beta values for beta distribution.
  vector[nBandits] initBETA;
  
  for (b in 1:nBandits)
  {initV[b] = payoff[b];} //is there a better way?
  initALPHA = rep_vector(1, nBandits); //initialise to uniform distribution
  initBETA = rep_vector(1, nBandits);

}

parameters {
  //Group level parameters
  vector[2] mu_pr;
  vector<lower=0>[2]sigma;
  
  
  //Subject level parameters reparameterised for matt trick
  // U = V + theta(mu + omega * sigma)
  vector[nSubjects] theta_pr; //bonus to MES stimuli
  vector[nSubjects] tau_pr; //inverse temp

}

transformed parameters{
  
  //Matt trick
  vector[nSubjects] theta;
  //vector[nSubjects] omega;
  vector<lower=0,upper=20>[nSubjects] tau;
  
  for (s in 1:nSubjects){
    theta[s] = mu_pr[1] + sigma[1] * theta_pr[s];
    tau[s] = Phi_approx(mu_pr[2] + sigma[2] * tau_pr[s]) * 20;
  }
}

model {
  //Hyperparameters
  mu_pr ~ normal(0,1);
  sigma ~ normal(0,1);
  
  //individual parameters
  theta_pr ~ normal(0,1);
  //omega_pr ~ normal(0,1);
  tau_pr ~ normal(0,1);
  
  for (s in 1:nSubjects){
    //Initialise Values
    vector[2] Q; // Q[1] = utility of optA, Q[2] = utility of optB
    vector[nBandits] alpha;
    vector[nBandits] beta;
    vector[nBandits] EP; //Expected probability of shock
    vector[nBandits] VAR; //Exoected variance
    int optA;
    int optB;
    int key;
    
    alpha = initALPHA;
    beta = initBETA;
    
    for (t in 1:nTrials){
      
      if (choice_stim[s,t] != 0) {
        for (b in 1:nBandits){
          EP[b] = alpha[b]/(alpha[b] + beta[b]);
          VAR[b] = sqrt((alpha[b] * beta[b]) / (((alpha[b]+beta[b])^2)*(alpha[b]+beta[b]+1)));
          }
      
        optA = option_A[s,t];
        optB = option_B[s,t]; //Grab the fractals presented at each trial (2AFC)!
      
        Q[1] = initV[optA] + theta[s] * EP[optA]; //+ omega[s] * VAR[optA]);
        Q[2] = initV[optB] + theta[s] * EP[optB]; //+ omega[s] * VAR[optB]);
      
        if (choice_stim[s,t] == optA) key = 1;
        else if (choice_stim[s,t] == optB) key = 2;
      
      
        key ~ categorical_logit(tau[s] * Q); // softmax prediction 
      
      //Update
        if (shock[s,t] == 1) alpha[choice_stim[s,t]] += 1;
        else beta[choice_stim[s,t]] += 1;
      }
    }
  }
}

generated quantities {
  // for group level parameters
  real mu_theta;
  //real mu_omega;
  real mu_tau;
  
  // Log-likelihood calculation
  real log_lik[nSubjects];
  
  // For PPC
  real y_pred[nSubjects, nTrials];

  // Set all PPs to -1
  for (s in 1:nSubjects){
    for (t in 1:nTrials){
      y_pred[s,t] = -1;
    }
  }
  
  mu_theta = mu_pr[1];
  //mu_omega = mu_pr[2];
  mu_tau = Phi_approx(mu_pr[2]) * 20;
  
  { // local section 
    for (s in 1:nSubjects){
      //Initialise Values
      vector[2] Q; // Q[1] = utility of optA, Q[2] = utility of optB
      vector[nBandits] alpha;
      vector[nBandits] beta;
      vector[nBandits] EP; //Expected probability of shock
      vector[nBandits] VAR; //Expected variance
      int optA;
      int optB;
      int key;
    
      log_lik[s] = 0;
      
      alpha = initALPHA;
      beta = initBETA;
    
      for (t in 1:nTrials){
        
        if (choice_stim[s,t] != 0) {
        
        for (b in 1:nBandits){
          EP[b] = alpha[b]/(alpha[b] + beta[b]);
          VAR[b] = sqrt((alpha[b] * beta[b]) / (((alpha[b]+beta[b])^2)*(alpha[b]+beta[b]+1)));
          }
          
        optA = option_A[s,t];
        optB = option_B[s,t]; //Grab the fractals presented at each trial (2AFC)!
      
        Q[1] = initV[optA] + theta[s] * EP[optA]; //+ omega[s] * VAR[optA]);
        Q[2] = initV[optB] + theta[s] * EP[optB]; //+ omega[s] * VAR[optB]);
        
        if (choice_stim[s,t] == optA) key = 1;
        else if (choice_stim[s,t] == optB) key = 2;
      
        //Compute log_likelihood
        log_lik[s] += categorical_logit_lpmf(key | tau[s] * Q);
      
        //generate posterior prediction for PPC 
        y_pred[s,t] = categorical_rng(softmax(tau[s] * Q));
        
        //update
        if (shock[s,t] == 1) alpha[choice_stim[s,t]] += 1;
        else beta[choice_stim[s,t]] += 1;
        }
      }
    }
  }
}
