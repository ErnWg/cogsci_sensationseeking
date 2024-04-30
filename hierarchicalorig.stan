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
  vector[nBandits] MES; //initialise shcok probabilites
  
  
  for (b in 1:nBandits)
  {initV[b] = payoff[b];} //is there a better way?
  MES = [1,0,1,0,0,1,0,1]'; // 1 = shock ; 0 = no shock based on identity so no learning involved
 

}

parameters {
  //Group level parameters
  vector[2] mu_pr;
  vector<lower=0>[2]sigma;
  
  
  //Subject level parameters reparameterised for matt trick
  vector[nSubjects] theta_pr; //bonus to MES stimuli
  vector[nSubjects] tau_pr; //inverse temp
  //vector[nSubjects] alpha_pr; //alpha learning rate
}

transformed parameters{
  
  //Matt trick
  vector[nSubjects] theta;
  vector<lower=0,upper=20>[nSubjects] tau;
  //vector<lower=0,upper=1>[nSubjects] alpha;
  
  for (s in 1:nSubjects){
    theta[s] = mu_pr[1] + sigma[1] * theta_pr[s];
    tau[s] = Phi_approx(mu_pr[2] + sigma[2] * tau_pr[s]) * 20;
    //alpha[s] = Phi_approx(mu_pr[3] + sigma[3] * alpha_pr[s]);
  }
}

model {
  //Hyperparameters
  mu_pr ~ normal(0,1);
  sigma ~ normal(0,1);
  
  //individual parameters
  theta_pr ~ normal(0,1);
  tau_pr ~ normal(0,1);
  //alpha_pr ~ normal(0,1);
  
  for (s in 1:nSubjects){
    //Initialise Values
    vector[2] Q; // Q[1] = utility of optA, Q[2] = utility of optB
    //vector[nBandits] EP; //Expected probability of shock
    int optA;
    int optB;
    int key;
    
    //EP = initShock;
    
    for (t in 1:nTrials){
      
      if (choice_stim[s,t] != 0) {
      
        optA = option_A[s,t];
        optB = option_B[s,t]; //Grab the fractals presented at each trial (2AFC)!
      
        Q[1] = initV[optA] + theta[s] * MES[optA]; //+ omega[s] * VAR[optA];
        Q[2] = initV[optB] + theta[s] * MES[optB]; //+ omega[s] * VAR[optB];
      
        if (choice_stim[s,t] == optA) key = 1;
        else if (choice_stim[s,t] == optB) key = 2;
      
      
        key ~ categorical_logit(tau[s] * Q);
      
      }
    }
  }
}

generated quantities {
  // for group level parameters
  real mu_theta;
  real<lower=0,upper=20> mu_tau;
  //real<lower=0,upper=1> mu_alpha;
  
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
  mu_tau = Phi_approx(mu_pr[2]) * 20;
  //mu_alpha = Phi_approx(mu_pr[3]);
  
  { // local section 
    for (s in 1:nSubjects){
      //Initialise Values
      vector[2] Q; // Q[1] = utility of optA, Q[2] = utility of optB
      int optA;
      int optB;
      int key;
    
      log_lik[s] = 0;
    
      for (t in 1:nTrials){
        
        if (choice_stim[s,t] != 0) {
      
        optA = option_A[s,t];
        optB = option_B[s,t]; //Grab the fractals presented at each trial (2AFC)!
      
        Q[1] = initV[optA] + theta[s] * MES[optA]; //+ omega[s] * VAR[optA];
        Q[2] = initV[optB] + theta[s] * MES[optB]; //+ omega[s] * VAR[optB];
        
        if (choice_stim[s,t] == optA) key = 1;
        else if (choice_stim[s,t] == optB) key = 2;
      
        //Compute log_likelihood
        log_lik[s] += categorical_logit_lpmf(key | tau[s] * Q);
      
        //generate posterior prediction for PPC 
        y_pred[s,t] = categorical_rng(softmax(tau[s] * Q));
      
        }
      }
    }
  }
}
