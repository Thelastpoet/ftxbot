---

# Nobuk System Design Document (SDD)

**Project:** Visitor & Access Control SaaS Platform
**Backend:** Spring Boot 3 + Spring Security + Spring Data JPA (Hibernate)
**Database:** PostgreSQL (Dockerized)
**Frontend:** Next.js (Admin Dashboard), React Native (Mobile App)

---

## 1. Architecture Overview

* **Pattern:** Modular Monolith (Spring Boot services organized by domain).
* **Tenancy Strategy:**

  * SaaS: **Row-Level Multi-Tenancy** (`tenantId` on all records).
  * On-Prem: Single-tenant, dedicated DB instance (same codebase, different config).
* **APIs:** REST (JSON), documented via **OpenAPI/Swagger**.
* **Auth:** JWT (stateless), managed by Spring Security.
* **Deployment:** Docker Compose (dev/on-prem), Kubernetes (SaaS).

---

## 2. Core Modules

1. **Authentication & Authorization**

   * Spring Security + JWT.
   * Roles: `GUARD`, `RESIDENT`, `TENANT_ADMIN`, `SUPER_ADMIN`.
   * Multi-tenancy enforced via `tenantId` in JWT claims.

2. **Tenant Management**

   * Create, update, delete estates (tenants).
   * Subscription & billing status stored here.

3. **User Management**

   * Residents: linked to `house/unit`.
   * Guards: assigned to tenant.
   * SaaS Super Admin: global scope.

4. **Visitor Management**

   * VisitorLog (check-in/out).
   * Exit code generation & validation.
   * Status: `IN`, `OUT`, `DENIED`.

5. **Notification Service**

   * Integrates with bulk SMS provider (REST API).
   * Logs delivery status in `SMSLog`.
   * Retry failed sends (Spring Retry).

6. **Audit & Logs**

   * Store all sensitive events (check-in/out, login attempts, failed code entries).
   * Retention policy configurable per tenant.

---

## 3. Database Design (Postgres ERD)

### Tables (simplified)

* **Tenant**

  * `id` (UUID, PK)
  * `name` (string)
  * `contact_email`
  * `contact_phone`
  * `status` (ACTIVE, SUSPENDED)
  * `created_at`, `updated_at`

* **User**

  * `id` (UUID, PK)
  * `tenant_id` (FK → Tenant)
  * `name`
  * `phone`
  * `role` (ENUM: GUARD, RESIDENT, TENANT\_ADMIN, SUPER\_ADMIN)
  * `house_number` (nullable, only for RESIDENT)
  * `password_hash`
  * `created_at`, `updated_at`

* **VisitorLog**

  * `id` (UUID, PK)
  * `tenant_id` (FK → Tenant)
  * `visitor_name`
  * `visitor_phone`
  * `vehicle_plate`
  * `purpose`
  * `checkin_time`
  * `checkout_time` (nullable)
  * `status` (IN, OUT, DENIED)
  * `exit_code` (unique, generated at check-in)
  * `created_by` (FK → User \[Guard])

* **SMSLog**

  * `id` (UUID, PK)
  * `tenant_id`
  * `recipient_phone`
  * `message_body`
  * `status` (SENT, FAILED, RETRIED)
  * `sent_at`

* **AuditLog**

  * `id` (UUID, PK)
  * `tenant_id`
  * `actor_id` (FK → User)
  * `action` (LOGIN, CHECKIN, CHECKOUT, CODE\_FAILED, etc.)
  * `metadata` (JSONB: request info, IP, etc.)
  * `timestamp`

* **Subscription**

  * `id` (UUID, PK)
  * `tenant_id`
  * `plan` (FREE, STANDARD, PREMIUM)
  * `sms_quota`
  * `billing_status` (ACTIVE, PAST\_DUE, CANCELLED)
  * `renewal_date`

---

## 4. API Endpoints (REST, JSON)

### Auth

* `POST /api/auth/login` → returns JWT.
* `POST /api/auth/register` → register new tenant admin (SaaS signup).

### Tenant Admin

* `GET /api/tenants/{id}`
* `PATCH /api/tenants/{id}`

### Users

* `POST /api/users` (create resident/guard)
* `GET /api/users?tenantId=...`
* `PATCH /api/users/{id}`
* `DELETE /api/users/{id}`

### Visitors

* `POST /api/visitors/checkin`
* `POST /api/visitors/checkout` (with exit code)
* `GET /api/visitors?tenantId=...&status=IN`
* `GET /api/visitors/{id}`

### Notifications

* `POST /api/notifications/sms` (send SMS manually if needed).
* `GET /api/notifications/logs`

### Audit

* `GET /api/audit?tenantId=...`

---

## 5. Multi-Tenancy Enforcement

* **Row-Level Security**: every entity includes `tenant_id`.
* Spring Data JPA filter auto-applied based on `tenantId` in JWT.
* Example:

```java
@PreAuthorize("hasRole('TENANT_ADMIN') and @tenantSecurity.checkTenantId(#tenantId)")
public List<User> getUsers(UUID tenantId) { ... }
```

* For **on-premise mode** → multi-tenancy filter disabled, single tenant only.

---

## 6. Code Structure (Spring Boot Packages)

```
src/main/java/com/securityapp/
 ├── auth/         # JWT, Spring Security config
 ├── tenants/      # Tenant mgmt, subscriptions
 ├── users/        # Residents, guards, admins
 ├── visitors/     # Visitor checkin/out, codes
 ├── notifications/# SMS service integration
 ├── audit/        # Audit logging
 ├── common/       # Shared utils, constants, base entities
 └── config/       # App config, multi-tenancy filter
```

---

## 7. Non-Functional Requirements

* **Performance:**

  * Check-in API < 500ms response.
  * DB queries optimized with indexes (`tenant_id`, `status`, `exit_code`).

* **Reliability:**

  * Retry SMS on failure (max 3 attempts).
  * Offline-first mobile app sync.

* **Scalability:**

  * SaaS mode scales horizontally (Spring Boot stateless + Postgres + Redis/Kafka).

* **Monitoring:**

  * Spring Boot Actuator endpoints `/actuator/health`, `/metrics`.
  * Prometheus + Grafana for dashboards.

* **Compliance:**

  * Logs retention policy (configurable per tenant).
  * Data export/delete endpoints for GDPR/DPA compliance.

---

## 8. Development Tools & Practices

* **Build:** Maven or Gradle.
* **Docs:** Swagger/OpenAPI via `springdoc-openapi`.
* **Testing:** JUnit 5 + Testcontainers (for Postgres in CI).
* **CI/CD:** GitHub Actions → build, test, Docker image.
* **Versioning:** Semantic Versioning (e.g., `v1.0.0`).

---