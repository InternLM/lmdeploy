use crate::routes::interface::RouteHandle;
use crate::routes::{SingleServerRoute, round_robin_route::RoundRobinRoute};
use crate::types::ConfigurationError;
use crate::utils::json::RequireField;
use serde_json::{Map, Value};

pub struct RoutingTreeBuilder {
    config: String,
}

impl RoutingTreeBuilder {
    pub fn new(config: String) -> Self {
        Self { config }
    }

    pub fn from_file(config: &std::path::PathBuf) -> Result<Self, ConfigurationError> {
        let json = std::fs::read_to_string(config)
            .map_err(|e| ConfigurationError(format!(
                "Failed to read config file '{}': {}",
                config.display(),
                e
            )))?;
        Ok(RoutingTreeBuilder::new(json))
    }

    pub fn build_routing_tree(
        self,
    ) -> Result<Box<dyn RouteHandle + Send + Sync>, ConfigurationError> {
        let json: Value = serde_json::from_str(&self.config)
            .map_err(|e| ConfigurationError(format!(
                "Invalid JSON config: {}",
                e
            )))?;
        let route: &Map<String, Value> = json.require("route")?;

        let root: Box<dyn RouteHandle + Send + Sync> = match route.require::<&str>("type")? {
            "SingleServerRoute" => Box::new(SingleServerRoute::new(
                route.require("host")?,
                route.require("port")?,
            )),
            "RoundRobinRoute" => {
                let servers = route
                    .require::<&Vec<Value>>("servers")?
                    .iter()
                    .map(|value| {
                        let host = value.require::<&str>("host")?;
                        let port = value.require::<u16>("port")?;

                        Ok((host, port))
                    })
                    .collect::<Result<Vec<(&str, u16)>, ConfigurationError>>()?;
                Box::new(RoundRobinRoute::new(servers))
            }
            unsupported_type => {
                return Err(ConfigurationError(format!(
                    "Unsupported route type: {}",
                    unsupported_type
                )));
            }
        };

        Ok(root)
    }
}
