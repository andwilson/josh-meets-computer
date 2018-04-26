import React from "react";
import Link from "gatsby-link";
import Helmet from "react-helmet";
import styled from "styled-components";
import Img from "gatsby-image";

import CategoryHeader from "../components/CategoryHeader";

const SLink = styled(Link)`
  text-decoration: none;
  color: #28aa55;
  transition: all 0.2s;
  &:hover {
    color: #1d7f3f;
  }
`;

class Projects extends React.Component {
  render() {
    const posts = this.props.data.allMarkdownRemark.edges;
    const categoryTitle = "Projects";
    const categoryDescription =
      "Below highlight a collection of projects I've worked on. A few years ago I decided to make Artificial Intelligence my single focus as an engineer, and recently refined my focused on applying AI/ML algorithms and systems to relationships and experiences. A bit of a catch-all for my ventures in this space!";

    return (
      <div>
        <Helmet title={categoryTitle} />
        <CategoryHeader
          title={categoryTitle}
          description={categoryDescription}
          data={this.props.data}
        />
        {posts.map(post => {
          if (post.node.path !== "/404/") {
            return (
              <div key={post.node.frontmatter.path}>
                <h3>
                  <SLink to={post.node.frontmatter.path}>
                    {post.node.frontmatter.title}
                  </SLink>
                </h3>
                <p>{post.node.frontmatter.date}</p>
                <p>{post.node.frontmatter.thumbnail.id}</p>
                <Img
                  sizes={post.node.frontmatter.thumbnail.childImageSharp.sizes}
                />
                <p dangerouslySetInnerHTML={{ __html: post.node.excerpt }} />
              </div>
            );
          }
        })}
      </div>
    );
  }
}

export default Projects;

export const pageQuery = graphql`
  query ProjectsQuery {
    allMarkdownRemark(
      sort: { fields: [frontmatter___date], order: DESC }
      filter: { frontmatter: { category: { eq: "Projects" } } }
    ) {
      totalCount
      edges {
        node {
          excerpt(pruneLength: 300)
          frontmatter {
            path
            date(formatString: "MMMM D, YYYY")
            title
            category
            thumbnail {
              childImageSharp {
                sizes(maxWidth: 800) {
                  ...GatsbyImageSharpSizes
                }
              }
            }
          }
        }
      }
    }
    avatar: imageSharp(id: { regex: "/avatar.jpg/" }) {
      sizes(maxWidth: 500, grayscale: false) {
        ...GatsbyImageSharpSizes_tracedSVG
      }
    }
  }
`;
