import flwr as fl
from flwr.common import FitIns  
from utils import load_original_text, aggregate_safe_summaries
from flwr.common import Parameters

# Global storage for per-round text
ROUND_TEXT = {}


def get_global_text(rnd: int) -> str:
    """
    It retrieve the text assigned to a given round.

    """
    return ROUND_TEXT.get(rnd, "")

class SafeFedAvg(fl.server.strategy.FedAvg):
    """
    Custom federated averaging strategy that injects global text into
    Fit instructions and aggregates safe summaries at the server level.
    """

    def configure_fit(self, server_round: int, parameters, client_manager):
        """
        Inject round number and global text into each client's FitIns config.
        """
        # Config sent to each client containing text to summarise
        config = {
            "round": server_round,
            "global_text": get_global_text(server_round)
        }
        
        # Get default instructions from parent FedAvg
        fit_ins_list = super().configure_fit(
            server_round=server_round,
            parameters=parameters,
            client_manager=client_manager
        )
        
        # Merge our config into existing FitIns objects
        new_fit_ins = []
        for client_proxy, fit_ins in fit_ins_list:
            # Combine original and custom config using dict unpacking
            merged_config = {**fit_ins.config, **config}

            # Create a new FitIns containing merged config
            new_fit_ins_obj = FitIns(
                parameters=fit_ins.parameters,
                config=merged_config
            )
            
            new_fit_ins.append((client_proxy, new_fit_ins_obj))
            
        return new_fit_ins

    def aggregate_fit(self, server_round: int, results, failures):
        """
        Handles all client Fit results, performs server-side safety inspection,
        and prepares the global text for the next round.
        """
        print(f"\n[ROUND {server_round}] Server received: {len(results)} good | {len(failures)} failed")
        if failures:
            print(f"Round {server_round}: {len(failures)} client failures")

        # Extract client summaries or metrics for safety processing
        client_summaries = []
        for client, fit_res in results:
            client_summaries.append((client, fit_res.metrics, None))

        # Get the original source text for this round
        source_text = get_global_text(server_round)

        # Run safety aggregation and obtain the next round's text
        next_text = aggregate_safe_summaries(
            client_summaries,
            source_text,
            current_round=server_round
        )

        # Store aggregated output as next round's input
        ROUND_TEXT[server_round + 1] = next_text

        # Return empty parameters as summarisation system does not update model
        return Parameters(tensors=[], tensor_type=""), {}


if __name__ == "__main__":
    """
    It a entry point for starting the Flower federated server.
    Loads initial text, configures strategy and starts training.
    """
    # Load the input text for round 1
    ROUND_TEXT[1] = load_original_text()

    # Custom strategy ensuring safety aware summarisation workflow
    strategy = SafeFedAvg(
        fraction_fit=1.0,
        min_fit_clients=5,
        min_available_clients=5,
    )

    # Start Flower server with two rounds
    fl.server.start_server(
        server_address="0.0.0.0:8080",
        config=fl.server.ServerConfig(num_rounds=2),
        strategy=strategy,
    )
