// Matchroom on Azure: Container Apps + managed identity + Application Insights.
// Deployed by `azd up`. No API keys: the app calls the existing Foundry model with Entra ID.
targetScope = 'resourceGroup'

@description('azd environment name, used to name resources')
param environmentName string

@description('Region for the new resources')
param location string = resourceGroup().location

@description('Name of the EXISTING Foundry / Azure OpenAI resource in this resource group, e.g. matchroom-resource')
param aiResourceName string

@description('Model deployment name in Foundry')
param modelDeployment string = 'gpt-5.4-mini'

@description('Azure OpenAI API version')
param apiVersion string = '2025-04-01-preview'

@description('Set by azd during deploy; leave the default on first provision')
param webImageName string = ''

var token = toLower(uniqueString(subscription().id, resourceGroup().id, environmentName))
var tags = { 'azd-env-name': environmentName, app: 'matchroom' }
var openAiUserRoleId = '5e0bd9bd-7b93-4f28-af87-19fc36ad61bd' // Cognitive Services OpenAI User
var acrPullRoleId = '7f951dda-4ed3-4680-a7ca-43fe172d538d'     // AcrPull

resource ai 'Microsoft.CognitiveServices/accounts@2024-10-01' existing = {
  name: aiResourceName
}

resource logs 'Microsoft.OperationalInsights/workspaces@2023-09-01' = {
  name: 'log-matchroom-${token}'
  location: location
  tags: tags
  properties: { sku: { name: 'PerGB2018' }, retentionInDays: 30 }
}

resource appInsights 'Microsoft.Insights/components@2020-02-02' = {
  name: 'appi-matchroom-${token}'
  location: location
  tags: tags
  kind: 'web'
  properties: { Application_Type: 'web', WorkspaceResourceId: logs.id }
}

resource identity 'Microsoft.ManagedIdentity/userAssignedIdentities@2023-01-31' = {
  name: 'id-matchroom-${token}'
  location: location
  tags: tags
}

resource registry 'Microsoft.ContainerRegistry/registries@2023-07-01' = {
  name: 'crmatchroom${token}'
  location: location
  tags: tags
  sku: { name: 'Basic' }
  properties: { adminUserEnabled: false }
}

resource acrPull 'Microsoft.Authorization/roleAssignments@2022-04-01' = {
  name: guid(registry.id, identity.id, acrPullRoleId)
  scope: registry
  properties: {
    principalId: identity.properties.principalId
    principalType: 'ServicePrincipal'
    roleDefinitionId: subscriptionResourceId('Microsoft.Authorization/roleDefinitions', acrPullRoleId)
  }
}

resource openAiUser 'Microsoft.Authorization/roleAssignments@2022-04-01' = {
  name: guid(ai.id, identity.id, openAiUserRoleId)
  scope: ai
  properties: {
    principalId: identity.properties.principalId
    principalType: 'ServicePrincipal'
    roleDefinitionId: subscriptionResourceId('Microsoft.Authorization/roleDefinitions', openAiUserRoleId)
  }
}

resource env 'Microsoft.App/managedEnvironments@2024-03-01' = {
  name: 'cae-matchroom-${token}'
  location: location
  tags: tags
  properties: {
    appLogsConfiguration: {
      destination: 'log-analytics'
      logAnalyticsConfiguration: {
        customerId: logs.properties.customerId
        sharedKey: logs.listKeys().primarySharedKey
      }
    }
  }
}

resource web 'Microsoft.App/containerApps@2024-03-01' = {
  name: 'ca-matchroom-${token}'
  location: location
  tags: union(tags, { 'azd-service-name': 'web' })
  identity: {
    type: 'UserAssigned'
    userAssignedIdentities: { '${identity.id}': {} }
  }
  dependsOn: [ acrPull ]
  properties: {
    managedEnvironmentId: env.id
    configuration: {
      ingress: { external: true, targetPort: 8000, transport: 'auto' }
      registries: [ { server: registry.properties.loginServer, identity: identity.id } ]
    }
    template: {
      containers: [
        {
          name: 'web'
          image: empty(webImageName) ? 'mcr.microsoft.com/azuredocs/containerapps-helloworld:latest' : webImageName
          resources: { cpu: json('0.5'), memory: '1Gi' }
          env: [
            { name: 'MATCHROOM_BACKEND', value: 'foundry' }
            { name: 'AZURE_OPENAI_ENDPOINT', value: 'https://${aiResourceName}.openai.azure.com/' }
            { name: 'AZURE_OPENAI_DEPLOYMENT', value: modelDeployment }
            { name: 'AZURE_OPENAI_API_VERSION', value: apiVersion }
            { name: 'AZURE_CLIENT_ID', value: identity.properties.clientId }
            { name: 'APPLICATIONINSIGHTS_CONNECTION_STRING', value: appInsights.properties.ConnectionString }
            { name: 'PORT', value: '8000' }
          ]
        }
      ]
      // one replica: the live match is held in memory and shared by every viewer
      scale: { minReplicas: 1, maxReplicas: 1 }
    }
  }
}

output AZURE_CONTAINER_REGISTRY_ENDPOINT string = registry.properties.loginServer
output AZURE_CONTAINER_REGISTRY_NAME string = registry.name
output WEB_URL string = 'https://${web.properties.configuration.ingress.fqdn}'
output APPLICATIONINSIGHTS_NAME string = appInsights.name
