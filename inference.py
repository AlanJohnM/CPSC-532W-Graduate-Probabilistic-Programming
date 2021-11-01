import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from likelihood_weighting import likelihood_weighting
from black_box_variational_inference import black_box_variational_inference
from metropolis_hastings_in_gibbs import metropolis_hastings_in_gibbs
from metropolis_hastings_in_gibbs import deterministic_eval as gibbs_eval
from metropolis_hastings_in_gibbs import burn as gibbs_burn
from hamiltonian_monte_carlo import hamiltonian_monte_carlo
from hamiltonian_monte_carlo import deterministic_eval as hmc_eval
from hamiltonian_monte_carlo import burn as hmc_burn
from daphne import daphne
from tqdm import tqdm


def log_joint_density(P_cal, state, eval):
    log_joint_density = torch.tensor(0.0)
    for v in state:
        log_joint_density = log_joint_density + eval(P_cal[v][1],state).observe(state[v])
    return log_joint_density


    
def aggregate_samples(samples, program, algorithm, graph=None, plot=False):
    if program == '1':
        if algorithm == 'likelihood_weighting':
            results = [m.item() for w,m in samples]
            weights = [w.item() for w,m in samples]
            
            mean = torch.dot(torch.softmax(torch.tensor(weights), dim = 0),torch.tensor(results))
            var = torch.dot(torch.softmax(torch.tensor(weights), dim = 0), (torch.tensor(results) - mean)**2)
            mean = mean.item()
            
            print(f"Weighted Expectation of Mu: {mean}")
            print(f"Weighted Variance of Mu: {var}")
            
            if plot:
                plt.hist(results, weights=[w.item() for w in torch.softmax(torch.tensor(weights), dim=0)], bins=100, range=(mean-5,mean+5))
                plt.title("Posterior for Mu via IS")
                plt.savefig("figures/1_hist_mu_IS.png")
                plt.clf()
            
        elif algorithm == 'mh_in_gibbs':
            results = [m.item() for w,m in samples]
            weights = [log_joint_density(graph[1]['P'], w, gibbs_eval).item() for w,m in samples]
            
            mean = torch.mean(torch.tensor(results[gibbs_burn:]))
            mean = mean.item()
            
            print(f"Expectation of Mu: {mean}")
            print(f"Variance of Mu: {torch.var(torch.tensor(results[gibbs_burn:]))}")
            
            if plot:
                
                plt.hist(results[gibbs_burn:], bins=100, range=(mean-5,mean+5))
                plt.title("Posterior for Mu via MH in Gibbs")
                plt.savefig("figures/1_hist_mu_mhg.png")
                plt.clf()
                
                plt.plot(weights)
                plt.title("Trace of log probability of Mu via MH in Gibbs")
                plt.savefig("figures/1_trace_log_w_mhg.png")
                plt.clf()
                
                plt.plot(results)
                plt.title("Trace of Mu via MH in Gibbs")
                plt.savefig("figures/1_trace_mu_mhg.png")
                plt.clf()

                
        elif algorithm == 'hmc':
            results = [m.item() for w,m in samples]
            weights = [log_joint_density(graph[1]['P'], w, hmc_eval).item() for w,m in samples]

            mean = torch.mean(torch.tensor(results[hmc_burn:]))
            mean = mean.item()
            print(f"Expectation of Mu: {mean}")
            print(f"Variance of Mu: {torch.var(torch.tensor(results[hmc_burn:]))}")

            if plot:
                plt.hist(results[hmc_burn:], bins=100, range=(mean-5,mean+5))
                plt.title("Posterior for Mu via HMC")
                plt.savefig("figures/1_hist_mu_hmc.png")
                plt.clf()
                
                plt.plot(weights)
                plt.title("Trace of log probability of Mu via HMC")
                plt.savefig("figures/1_trace_log_w_hmc.png")
                plt.clf()
                
                plt.plot(results)
                plt.title("Trace of Mu via MH in HMC")
                plt.savefig("figures/1_trace_mu_hmc.png")
                plt.clf()
                
        elif algorithm == 'bbvi':
            pass

            
    elif program == '2':
        params = ['Slope','Bias']
        if algorithm == 'likelihood_weighting':
            for i in range(2):
                results = [m[i].item() for w,m in samples]
                weights = [w.item() for w,m in samples]
                
                mean = torch.dot(torch.softmax(torch.tensor(weights), dim = 0),torch.tensor(results))
                mean = mean.item()

                print(f"Weighted Expectation of {params[i]}: {mean}")
                
                if plot:
                    plt.hist(results, weights=[w.item() for w in torch.softmax(torch.tensor(weights), dim=0)], bins=100, range=(mean-5,mean+5))
                    plt.title(f"Posterior for {params[i]} via IS")
                    plt.savefig(f"figures/2_hist_{params[i]}_IS.png")
                    plt.clf()
            
            slopes = np.array([m[0].item() for w,m in samples])
            biases = np.array([m[1].item() for w,m in samples])
            covariance = np.cov(np.array([slopes,biases]), ddof=0, aweights=np.array(torch.softmax(torch.tensor(weights), dim=0)))
            print(f"The Coviariance Matrix of slope and bias is: {covariance}")
            
        elif algorithm == 'mh_in_gibbs':
            for i in range(2):
                results = [m[i].item() for w,m in samples]
                weights = [log_joint_density(graph[1]['P'], w, gibbs_eval).item() for w,m in samples]
                
                mean = torch.mean(torch.tensor(results[gibbs_burn:]))
                mean = mean.item()
                print(f"Expectation of {params[i]}: {mean}")
                
                
                if plot:
                    plt.hist(results[gibbs_burn:], bins=100, range=(mean-10,mean+10))
                    plt.title(f"Posterior for {params[i]} via MH in Gibbs")
                    plt.savefig(f"figures/2_hist_{params[i]}_mhg.png")
                    plt.clf()
                    
                    plt.plot(weights)
                    plt.title(f"Trace of {params[i]} via MH in Gibbs")
                    plt.savefig(f"figures/2_trace_log_mhg.png")
                    plt.clf()
                    
                    plt.plot(results)
                    plt.title(f"Trace of log probability of {params[i]} via MH in Gibbs")
                    plt.savefig(f"figures/2_trace_{params[i]}_mhg.png")
                    plt.clf()
            
            slopes = np.array([m[0].item() for w,m in samples])
            biases = np.array([m[1].item() for w,m in samples])
            covariance = np.cov(np.array([slopes,biases]))
            print(f"The Coviariance Matrix of slope and bias is: {covariance}")

                
        elif algorithm == 'hmc':
            for i in range(2):
                results = [m[i].item() for w,m in samples]
                weights = [log_joint_density(graph[1]['P'], w, hmc_eval).item() for w,m in samples]

                mean = torch.mean(torch.tensor(results[hmc_burn:]))
                print(f"Expectation of {params[i]}: {mean}")
                
                if plot:
                    plt.hist(results[hmc_burn:], bins=100)
                    plt.title(f"Posterior for {params[i]} via HMC")
                    plt.savefig(f"figures/2_hist_{params[i]}_hmc.png")
                    plt.clf()
                    
                    plt.plot(results)
                    plt.title(f"Trace of {params[i]} via HMC")
                    plt.savefig(f"figures/2_trace_{params[i]}_hmc.png")
                    plt.clf()

                    plt.plot(weights)
                    plt.title(f"Trace of log probability of {params[i]} via HMC")
                    plt.savefig(f"figures/2_trace_log_w_hmc.png")
                    plt.clf()
                                
            slopes = np.array([m[0].item() for w,m in samples])
            biases = np.array([m[1].item() for w,m in samples])
            covariance = np.cov(np.array([slopes,biases]))
            print(f"The Coviariance Matrix of slope and bias is: {covariance}")
        
        elif algorithm == 'bbvi':
            pass

    elif program == '3':
        if algorithm == 'likelihood_weighting':
            results = [m.item() for w,m in samples]
            weights = [w.item() for w,m in samples]
            
            mean = torch.dot(torch.softmax(torch.tensor(weights), dim = 0),torch.tensor(results))
            var = torch.dot(torch.softmax(torch.tensor(weights), dim = 0), (torch.tensor(results) - mean)**2)
            
            print(f"Weighted Expectation of equality: {mean}")
            print(f"Weighted Variance of equality: {var}")
        
            if plot:
                plt.hist(results, weights=[w.item() for w in torch.softmax(torch.tensor(weights), dim=0)], bins=100)
                plt.title("Posterior for equality via IS")
                plt.savefig("figures/3_hist_equality_IS.png")
                plt.clf()
            
        elif algorithm == 'mh_in_gibbs':
            results = [m.item() for w,m in samples]
            weights = [log_joint_density(graph[1]['P'], w, gibbs_eval).item() for w,m in samples]
            
            
            print(f"Expectation of equality: {torch.mean(torch.tensor(results[gibbs_burn:]))}")
            print(f"Variance of equality: {torch.var(torch.tensor(results[gibbs_burn:]))}")
            
            if plot:
                plt.hist(results[gibbs_burn:], bins=100)
                plt.title("Posterior for equality via MH in Gibbs")
                plt.savefig("figures/3_hist_equality_mhg.png")
                plt.clf()
                
                plt.plot(weights)
                plt.title("Trace of log probability of equality via MH in Gibbs")
                plt.savefig("figures/3_trace_log_w_mhg.png")
                plt.clf()
                
                plt.plot(results)
                plt.title("Trace of equality via MH in Gibbs")
                plt.savefig("figures/3_trace_equality_mhg.png")
                plt.clf()

        elif algorithm == 'bbvi':
            pass


        
    elif program == '4':
        if algorithm == 'likelihood_weighting':
            results = [m.item() for w,m in samples]
            weights = [w.item() for w,m in samples]
            
            mean = torch.dot(torch.softmax(torch.tensor(weights), dim = 0),torch.tensor(results))
            var = torch.dot(torch.softmax(torch.tensor(weights), dim = 0), (torch.tensor(results) - mean)**2)

            print(f"Weighted Expectation of rain: {mean}")
            print(f"Weighted Variance of rain: {var}")
            
            if plot:
                plt.hist(results, weights=[w.item() for w in torch.softmax(torch.tensor(weights), dim=0)], bins=100)
                plt.title("Posterior for rain via IS")
                plt.savefig("figures/4_hist_rain_IS.png")
                plt.clf()
            
        elif algorithm == 'mh_in_gibbs':
            results = [m.item() for w,m in samples]
            weights = [log_joint_density(graph[1]['P'], w, gibbs_eval).item() for w,m in samples]
            
            
            print(f"Expectation of rain: {torch.mean(torch.tensor(results[gibbs_burn:]))}")
            print(f"Variance of rain: {torch.var(torch.tensor(results[gibbs_burn:]))}")
            
            if plot:
                plt.hist(results[gibbs_burn:], bins=100)
                plt.title("Posterior for rain via MH in Gibbs")
                plt.savefig("figures/4_hist_rain_mhg.png")
                plt.clf()
                
                plt.plot(weights)
                plt.title("Trace of log probability of rain via MH in Gibbs")
                plt.savefig("figures/4_trace_log_w_mhg.png")
                plt.clf()
                
                plt.plot(results)
                plt.title("Trace of rain via MH in Gibbs")
                plt.savefig("figures/4_trace_rain_mhg.png")
                plt.clf()
            
        elif algorithm == 'bbvi':
            pass


    elif program == '5':
        params = ['x','y']
        for i in range(2):
            if algorithm == 'likelihood_weighting':
                results = [m[i].item() for w,m in samples]
                weights = [w.item() for w,m in samples]
                
                mean = torch.dot(torch.softmax(torch.tensor(weights), dim = 0),torch.tensor(results))
                var = torch.dot(torch.softmax(torch.tensor(weights), dim = 0), (torch.tensor(results) - mean)**2)

                print(f"Weighted Expectation of {params[i]}: {mean}")
                print(f"Weighted Variance of {params[i]}: {var}")

                if plot:
                    plt.hist(results, weights=[w.item() for w in torch.softmax(torch.tensor(weights), dim=0)], bins=100)
                    plt.title(f"Posterior for {params[i]} via IS")
                    plt.savefig(f"figures/5_hist_{params[i]}_IS.png")
                    plt.clf()
                
            elif algorithm == 'mh_in_gibbs':
                results = [m[i].item() for w,m in samples]
                weights = [log_joint_density(graph[1]['P'], w, gibbs_eval).item() for w,m in samples]
                
                
                print(f"Expectation of {params[i]}: {torch.mean(torch.tensor(results))}")
                print(f"Variance of {params[i]}: {torch.var(torch.tensor(results))}")
                
                
                if plot:
                    plt.hist(results, bins=100)
                    plt.title(f"Posterior for {params[i]} via MH in Gibbs")
                    plt.savefig(f"figures/5_hist_{params[i]}_mhg.png")
                    plt.clf()
                    
                    plt.plot(results)
                    plt.title(f"Trace of {params[i]} via MH in Gibbs")
                    plt.savefig(f"figures/5_trace_{params[i]}_mhg.png")
                    plt.clf()
                    
                    plt.plot(weights)
                    plt.title(f"Trace of log probability of {params[i]} via MH in Gibbs")
                    plt.savefig(f"figures/5_trace_log_w_mhg.png")
                    plt.clf()
                
            elif algorithm == 'hmc':
                results = [m[i].item() for w,m in samples]
                weights = [log_joint_density(graph[1]['P'], w, hmc_eval).item() for w,m in samples]

                print(f"Expectation of {params[i]}: {torch.mean(torch.tensor(results))}")
                print(f"Variance of {params[i]}: {torch.var(torch.tensor(results))}")
                
                if plot:
                    plt.hist(results, bins=100)
                    plt.title(f"Posterior for {params[i]} via HMC")
                    plt.savefig(f"figures/5_hist_{params[i]}_hmc.png")
                    plt.clf()
                    
                    plt.plot(results)
                    plt.title(f"Trace of {params[i]} via HMC")
                    plt.savefig(f"figures/5_trace_{params[i]}_hmc.png")
                    plt.clf()

                    plt.plot(weights)
                    plt.title(f"Trace of log probability of {params[i]} via HMC")
                    plt.savefig(f"figures/5_trace_log_w_hmc.png")
                    plt.clf()
    
    
            elif algorithm == 'bbvi':
                pass

algorithms = {
        'likelihood_weighting':likelihood_weighting,
        'mh_in_gibbs':metropolis_hastings_in_gibbs,
        'hmc':hamiltonian_monte_carlo,
        'bbvi':black_box_variational_inference
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-L', '--n_samples', required=False, default=1000, type=int)
    parser.add_argument('-P', '--programs', required=False, default=['1','2','3','4','5'], type=str, nargs='*')
    parser.add_argument('-A', '--algorithm', required=False, default='bbvi', type=str)
    parser.add_argument('--plot', required=False,action='store_true' ,default=False)
    args = parser.parse_args()
    
    if args.algorithm not in algorithms.keys():
        raise Error("Unknown Inference Algorithm Selected")
    evaluator = 'desugar' if args.algorithm in ['likelihood_weighting', 'bbvi'] else 'graph'
    inference_algorithm = algorithms[args.algorithm]
    
    for p in args.programs:
        program = daphne([f'{evaluator}', '-i', f'../CS532-HW4/programs/{p}.daphne'])
        samples = inference_algorithm(program, args.n_samples)
        aggregate_samples(samples, p, args.algorithm, program, args.plot)
            
