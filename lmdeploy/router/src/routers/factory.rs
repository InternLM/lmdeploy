//! Factory for creating router instances

use super::{
    http::{lmdeploy_pd_router::LMDeployPDRouter, openai_router::OpenAIRouter, router::Router},
    RouterTrait,
};
use crate::config::{LMDeployMigrationProtocol, LMDeployRdmaConfig, PolicyConfig, RoutingMode};
use crate::policies::PolicyFactory;
use crate::server::AppContext;
use std::sync::Arc;

/// Static inputs needed to build an LMDeploy PD router.
pub struct LMDeployPDRouterParams<'a> {
    pub prefill_urls: &'a [String],
    pub decode_urls: &'a [String],
    pub migration_protocol: LMDeployMigrationProtocol,
    pub rdma_config: Option<LMDeployRdmaConfig>,
    pub dummy_prefill: bool,
    pub prefill_policy_config: Option<&'a PolicyConfig>,
    pub decode_policy_config: Option<&'a PolicyConfig>,
    pub main_policy_config: &'a PolicyConfig,
    pub ctx: &'a Arc<AppContext>,
}

/// Factory for creating router instances based on configuration
pub struct RouterFactory;

impl RouterFactory {
    /// Create a router instance from application context
    pub async fn create_router(ctx: &Arc<AppContext>) -> Result<Box<dyn RouterTrait>, String> {
        match &ctx.router_config.mode {
            RoutingMode::Regular { worker_urls } => {
                Self::create_regular_router(worker_urls, ctx).await
            }
            RoutingMode::LMDeployPrefillDecode {
                prefill_urls,
                decode_urls,
                prefill_policy,
                decode_policy,
                migration_protocol,
                rdma_config,
                dummy_prefill,
            } => {
                tracing::info!(
                    "Creating LMDeployPDRouter with prefill_urls: {:?}, decode_urls: {:?}, migration_protocol: {:?}",
                    prefill_urls, decode_urls, migration_protocol
                );
                Self::create_lmdeploy_pd_router(LMDeployPDRouterParams {
                    prefill_urls,
                    decode_urls,
                    migration_protocol: *migration_protocol,
                    rdma_config: rdma_config.clone(),
                    dummy_prefill: *dummy_prefill,
                    prefill_policy_config: prefill_policy.as_ref(),
                    decode_policy_config: decode_policy.as_ref(),
                    main_policy_config: &ctx.router_config.policy,
                    ctx,
                })
                .await
            }
            RoutingMode::OpenAI { worker_urls, .. } => {
                Self::create_openai_router(worker_urls.clone(), ctx).await
            }
        }
    }

    /// Create a regular router
    pub async fn create_regular_router(
        worker_urls: &[String],
        ctx: &Arc<AppContext>,
    ) -> Result<Box<dyn RouterTrait>, String> {
        // Create regular router with context
        let router = Router::new(worker_urls.to_vec(), ctx).await?;

        Ok(Box::new(router))
    }

    /// Create an LMD (lmdeploy) PD router with static URLs
    pub async fn create_lmdeploy_pd_router(
        params: LMDeployPDRouterParams<'_>,
    ) -> Result<Box<dyn RouterTrait>, String> {
        let LMDeployPDRouterParams {
            prefill_urls,
            decode_urls,
            migration_protocol,
            rdma_config,
            dummy_prefill,
            prefill_policy_config,
            decode_policy_config,
            main_policy_config,
            ctx,
        } = params;
        let prefill_policy =
            PolicyFactory::create_from_config(prefill_policy_config.unwrap_or(main_policy_config));
        let decode_policy =
            PolicyFactory::create_from_config(decode_policy_config.unwrap_or(main_policy_config));

        ctx.policy_registry.set_prefill_policy(prefill_policy);
        ctx.policy_registry.set_decode_policy(decode_policy);

        tracing::info!(
            "Creating LMDeployPDRouter with static URLs - prefill: {:?}, decode: {:?}",
            prefill_urls,
            decode_urls
        );

        let router = LMDeployPDRouter::new(
            prefill_urls.to_vec(),
            decode_urls.to_vec(),
            migration_protocol,
            rdma_config,
            dummy_prefill,
            ctx,
        )
        .await?;
        tracing::info!("LMDeployPDRouter instance created successfully");

        Ok(Box::new(router))
    }

    /// Create an OpenAI router
    async fn create_openai_router(
        worker_urls: Vec<String>,
        ctx: &Arc<AppContext>,
    ) -> Result<Box<dyn RouterTrait>, String> {
        // Use the first worker URL as the OpenAI-compatible base
        let base_url = worker_urls
            .first()
            .cloned()
            .ok_or_else(|| "OpenAI mode requires at least one worker URL".to_string())?;

        let router =
            OpenAIRouter::new(base_url, Some(ctx.router_config.circuit_breaker.clone())).await?;

        Ok(Box::new(router))
    }
}
